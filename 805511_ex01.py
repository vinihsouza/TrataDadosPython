#!/usr/bin/env python

FILES_DIRECTORY = "dados"

import numpy as np
import pandas as pd
import os

if __name__ == '__main__':
    df_dataset = pd.read_csv(os.path.join(FILES_DIRECTORY, 'iris.csv'), sep=',', index_col=None) 

    print('Dados importados com sucesso!')



if __name__ == '__main__':
    display(df_dataset.head(n=10))


if __name__ == '__main__':
    df_dataset = df_dataset.drop(columns=['id_planta','cidade_origem'])
    display(df_dataset.head(n=10))

if __name__ == '__main__':
    idxRowNan = pd.isnull(df_dataset).any(1).to_numpy().nonzero()
    display(df_dataset.iloc[idxRowNan])

def trataFaltantes( df_dataset ):
    
    idxRowNan = pd.isnull(df_dataset).any(1).to_numpy().nonzero()
    noDuplicates = (df_dataset.classe.iloc[idxRowNan]).drop_duplicates()
    soma = [0,0,0]
    count = [0,0,0]
    for i in df_dataset.values:
        if (i[4] == noDuplicates.values[0]):
            if(pd.isna(i[1]) == True and pd.isna(i[2]) == True):
                _
            elif(pd.isna(i[1]) == True):
                soma[2] += i[2]
                count[2] += 1
            elif(pd.isna(i[2]) == True):
                soma[1] += i[1]
                count[1] += 1
            else:
                soma[1] += i[1]
                soma[2] += i[2]
                count[1] += 1
                count[2] += 1

    md_largura_sepala = soma[1]/count[1]
    md_comprimento_petala = soma[2]/count[2]
    df_dataset.largura_sepala = df_dataset.largura_sepala.fillna(md_largura_sepala)
    df_dataset.comprimento_petala = df_dataset.comprimento_petala.fillna(md_comprimento_petala)
    
    return df_dataset.round(6)

if __name__ == '__main__':
    df_dataset = trataFaltantes( df_dataset )
    
    print('\nAmostras que possuiam valores faltantes:')
    display(df_dataset.iloc[idxRowNan])

if __name__ == '__main__':
    df_duplicates = df_dataset[ df_dataset.duplicated(subset=['comprimento_sepala','largura_sepala','comprimento_petala','largura_petala'],keep=False)] 

    if len(df_duplicates)>0:
        print('\nAmostras redundantes ou inconsistentes:')
        display(df_duplicates)
    else:
        print('Não existem valores duplicados')


def delDuplicatas( df_dataset ):

    df_dataset = df_dataset.drop_duplicates()

    return df_dataset

if __name__ == '__main__':
    df_dataset = delDuplicatas( df_dataset )


if __name__ == '__main__':

    df_duplicates = df_dataset[ df_dataset.duplicated(subset=['comprimento_sepala','largura_sepala','comprimento_petala','largura_petala'],keep=False)] 

    if len(df_duplicates)>0:
        print('\nAmostras inconsistentes:')
        display(df_duplicates)
    else:
        print('Não existem mostras inconsistentes')


def delInconsistencias( df_dataset ):

    df_duplicates = df_dataset[ df_dataset.duplicated(subset=['comprimento_sepala','largura_sepala','comprimento_petala','largura_petala'],keep=False)] 
    df_dataset = df_dataset.drop(df_duplicates.index)
    
    return df_dataset

if __name__ == '__main__':
    df_dataset = delInconsistencias( df_dataset )

    df_duplicates = df_dataset[ df_dataset.duplicated(subset=['comprimento_sepala','largura_sepala','comprimento_petala','largura_petala'],keep=False)] 

    if len(df_duplicates)>0:
        display(df_duplicates)
    else:
        print('Não existem amostras redundantes ou inconsistentes')

if __name__ == '__main__':
    df_detalhes = df_dataset.describe()
    print(df_detalhes)

def normalizar(X):
    """
    Normaliza os atributos em X
    
    Esta função retorna uma versao normalizada de X onde o valor da
    média de cada atributo é igual a 0 e desvio padrao é igual a 1. Trata-se de
    um importante passo de pré-processamento quando trabalha-se com 
    métodos de aprendizado de máquina.
    """
    
    m, n = X.shape 
    X_norm = np.random.rand(m,n)
    mu = 0
    sigma = 1
    
    mu = np.mean(X, axis=0, dtype=np.float64)
    sigma = np.std(X, axis=0, ddof=1, dtype=np.float64)
    X_norm = ((X - mu) / sigma)
    
    return X_norm, mu, sigma

if __name__ == '__main__':
    X = df_dataset.iloc[:,0:-1].values
    X_norm, mu, sigma = normalizar(X)
    df_dataset.iloc[:,0:-1] = X_norm
    print('\nPrimeira amostra da base antes da normalização: [%2.4f %2.4f].' %(X[0,0],X[0,1]))
    print('\nApós a normalização, espera-se que a primeira amostra seja igual a: [-0.5747 0.1804].')
    print('\nPrimeira amostra da base apos normalização: [%2.4f %2.4f].' %(X_norm[0,0],X_norm[0,1]))

if __name__ == '__main__':
    df_detalhes = df_dataset.describe()    
    display(df_detalhes.round(8))

import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    df_dataset.boxplot(figsize=(15,7))
    plt.show()

if __name__ == '__main__':
    pd.plotting.scatter_matrix(df_dataset, figsize=(18,18))    
    plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    sns.pairplot(df_dataset, hue='classe', height=3.5)
    
    plt.show()

if __name__ == '__main__':

    sns.lmplot(x='comprimento_sepala', y='largura_sepala', data=df_dataset, 
               fit_reg=False,
               hue='classe')
    
    plt.title('Comprimento vs largura da sepala')
    plt.show()

if __name__ == '__main__':
    for atributo in df_dataset.columns[:-1]:
        plt.figure(figsize=(8,8))
    
        sns.boxplot(x="classe", y=atributo, data=df_dataset, whis=1.5)
    
        plt.show()

if __name__ == '__main__':
    for atributo in df_dataset.columns[:-1]:

        n, bins, patches = plt.hist(df_dataset[atributo].values,bins=10, color='red', edgecolor='black', linewidth=0.9)

        plt.title(atributo)

        plt.show()

if __name__ == '__main__':
    for atributo in df_dataset.columns[:-1]:

        densityplot = df_dataset[atributo].plot(kind='density')

        plt.title(atributo)

        plt.show()

def removeOutliers(df_dataset):

    df_datasetNew = df_dataset.drop(columns=['classe'])
    Q3 = np.percentile(df_datasetNew, 75, axis=0)
    Q1 = np.percentile(df_datasetNew, 25, axis=0)
    IQR = (Q3 - Q1) * 1.5
    min_quartile = (Q1 - IQR)
    max_quartile = ( Q3 + IQR)
    count = 0
    countArray = []
    for i in df_datasetNew.values :
        if (i[0] < min_quartile[0] or i[1] < min_quartile[1] or i[2] < min_quartile[2] or i[3] < min_quartile[3] or i[0] > max_quartile[0] or i[1] > max_quartile[1] or i[2] > max_quartile[2] or i[3] > max_quartile[3]):
            countArray.append(df_dataset.index[count])
        count += 1
    df_dataset = df_dataset.drop(countArray)    
    
    return df_dataset

if __name__ == '__main__':
    df_dataset = removeOutliers( df_dataset )
    df_dataset.boxplot(figsize=(15,7))
    plt.show()
    sns.pairplot(df_dataset, hue='classe', height=3.5);
    plt.show()

if __name__ == '__main__':
    display( df_dataset['classe'].value_counts().sort_index() )

    sns.countplot(x="classe", data=df_dataset)

    plt.show()

if __name__ == '__main__':
    X = df_dataset.iloc[:,:-1].values
    covariance = np.cov(X, rowvar=False)
    correlation = np.corrcoef(X, rowvar=False)
    print('Matriz de covariância: ')
    display(covariance)
    print('\n\nMatriz de correlação: ')
    display(correlation)

if __name__ == '__main__':
    df_covariance = df_dataset.cov()
    df_correlation = df_dataset.corr()
    print('Matriz de covariância: ')
    display(df_covariance)
    print('\n\nMatriz de correlação: ')
    display(df_correlation)

if __name__ == '__main__':
    sns.heatmap(df_covariance, 
            xticklabels=df_correlation.columns,
            yticklabels=df_correlation.columns)

    plt.title('Covariancia')
    plt.show()

    sns.heatmap(df_correlation, 
            xticklabels=df_correlation.columns,
            yticklabels=df_correlation.columns)

    plt.title('Correlacao')
    plt.show()


if __name__ == '__main__':
    df_dataset2 = pd.read_csv(os.path.join(FILES_DIRECTORY, 'data2.csv'), sep=',', index_col=None) 

if __name__ == '__main__':
    import pandas as pd
    df_dataset2 = pd.read_csv(os.path.join(FILES_DIRECTORY, 'data2.csv'), sep=',', index_col=None) 
    display(df_dataset2.head(n=7))

if __name__ == '__main__':
    import pandas as pd
    df_dataset2 = pd.read_csv(os.path.join(FILES_DIRECTORY, 'data2.csv'), sep=',', index_col=None) 
    display(df_dataset2.describe())
    
if __name__ == '__main__':   
    import pandas as pd
    import matplotlib.pyplot as plt
    df_dataset2 = pd.read_csv(os.path.join(FILES_DIRECTORY, 'data2.csv'), sep=',', index_col=None) 
    n, bins, patches = plt.hist(df_dataset2['atributo_d'].values,bins=10, color='red', edgecolor='black', linewidth=0.9)
    plt.title('atributo_d')
    plt.show()

if __name__ == '__main__':
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    df_dataset2 = pd.read_csv(os.path.join(FILES_DIRECTORY, 'data2.csv'), sep=',', index_col=None) 
    sns.lmplot(x='atributo_a', y='atributo_b', data=df_dataset2,
               fit_reg=False,
               hue='classe')
    plt.title('atributo_a vs atributo_b')
    plt.show()

if __name__ == '__main__':
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    df_dataset2 = pd.read_csv(os.path.join(FILES_DIRECTORY, 'data2.csv'), sep=',', index_col=None) 
    display( df_dataset2['classe'].value_counts().sort_index())
    sns.countplot(x="classe", data=df_dataset2)
    plt.show()

if __name__ == '__main__':
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    df_dataset2 = pd.read_csv(os.path.join(FILES_DIRECTORY, 'data2.csv'), sep=',', index_col=None) 
    plt.figure(figsize=(8,8))
    sns.boxplot(x="classe", y='atributo_c', data=df_dataset2, whis=1.5)
    plt.show()

if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    
    df_dataset2 = pd.read_csv(os.path.join(FILES_DIRECTORY, 'data2.csv'), sep=',', index_col=None) 
    X = df_dataset2.iloc[:,:-1].values
    correlation = np.corrcoef(X, rowvar=False)
    print('\n\nMatriz de correlação: ')
    display(correlation)

def covariancia(atributo1, atributo2):  
    cov = 0 
    n = len(atributo1)

    import numpy as np
    xi = np.mean(atributo1)
    xj = np.mean(atributo2)
    newAtributo = (atributo1 - xi) * (atributo2 - xj)
    cov = (newAtributo.sum()) * (1/(n-1))

    return cov

if __name__ == '__main__':
    atributo1 = df_dataset2['atributo_a'].values
    atributo2 = df_dataset2['atributo_b'].values

    print('Valor esperado: 4.405083')

    cov = covariancia(atributo1, atributo2)
    print('Valor retornado pela função: %1.6f' %cov)

def correlacao(atributo1, atributo2):

    corr = 0 
    
    n = len(atributo1)

    import numpy as np
    xi = np.mean(atributo1)
    xj = np.mean(atributo2)
    newAtributo = (atributo1 - xi) * (atributo2 - xj)
    cov = (newAtributo.sum()) * (1/(n-1))
    dxi = np.std(atributo1, ddof=1)
    dxj = np.std(atributo2, ddof=1)
    corr = cov / (dxi * dxj)

    return corr

if __name__ == '__main__':
    atributo1 = df_dataset2['atributo_a'].values
    atributo2 = df_dataset2['atributo_b'].values

    print('Valor esperado: 0.264026')

    corr = correlacao(atributo1, atributo2)
    print('Valor retornado pela função: %1.6f' %corr)

