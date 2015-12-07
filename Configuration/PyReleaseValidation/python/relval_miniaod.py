# import the definition of the steps and input files:
from  Configuration.PyReleaseValidation.relval_steps import *

# here only define the workflows as a combination of the steps defined above:
workflows = Matrix()

# each workflow defines a name and a list of steps to be done. 
# if no explicit name/label given for the workflow (first arg),
# the name of step1 will be used


## re-miniAOD of the production tests
workflows[1501301] = ['', ['ProdMinBias_13_MINIAOD','REMINIAODPROD','HARVESTREMINIAODPROD']]
workflows[1501302] = ['', ['ProdTTbar_13_MINIAOD','REMINIAODPROD','HARVESTREMINIAODPROD']]
workflows[1501303] = ['', ['ProdQCD_Pt_3000_3500_13_MINIAOD','REMINIAODPROD','HARVESTREMINIAODPROD']]

## re-miniAOD workflows -- fullSim noPU
workflows[1501329] = ['', ['ZEE_13_REMINIAOD','REMINIAOD','HARVESTREMINIAOD']]
workflows[1501331] = ['', ['ZTT_13_REMINIAOD','REMINIAOD','HARVESTREMINIAOD']]
workflows[1501330] = ['', ['ZMM_13_REMINIAOD','REMINIAOD','HARVESTREMINIAOD']]

## re-miniAOD workflows -- fullSim PU
#workflows[50200]=['',['ZEE_13','DIGIUP15_PU50','RECOUP15_PU50','HARVESTUP15_PU50']]
#workflows[25200]=['',['ZEE_13','DIGIUP15_PU25','RECOUP15_PU25','HARVESTUP15_PU25']]

## re-miniAOD workflows -- data 2015b
workflows[150134.702] = ['',['RunDoubleEG2015B_MINIAOD','REMINIAODDR2_50ns','HARVESTREMINIAODDR2_50ns']]

## re-miniAOD workflows -- data 2015c
workflows[150134.802] = ['',['RunDoubleEG2015C_MINIAOD','REMINIAODDR2_25ns','HARVESTREMINIAODDR2_25ns']]

## re-miniAOD workflows -- data 2015d
workflows[150134.902] = ['',['RunDoubleEG2015D_MINIAOD','REMINIAODDR2_25ns','HARVESTREMINIAODDR2_25ns']]

