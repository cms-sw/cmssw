
# import the definition of the steps and input files:
from  Configuration.PyReleaseValidation.relval_steps import *

# here only define the workflows as a combination of the steps defined above:
workflows = Matrix()

# each workflow defines a name and a list of steps to be done. 
# if no explicit name/label given for the workflow (first arg),
# the name of step1 will be used



#just define all of them

numWFStart=10000
numWFSkip=200
#2017 WFs to run in IB (TenMuE_0_200, TTbar, ZEE, MinBias, TTbar PU, ZEE PU)
numWFIB = [10021.0,10024.0,10025.0,10026.0,10023.0,10224.0,10225.0]
numWFIB.extend([10821.0,10824.0,10825.0,10826.0]) #2023sim scenario
numWFIB.extend([10621.0,10624.0,10625.0,10626.0]) #2023 with tilted tracker
numWFIB.extend([11021.0,11024.0,11025.0,11026.0]) #2023lreco
for i,key in enumerate(upgradeKeys):
    numWF=numWFStart+i*numWFSkip
    for frag in upgradeFragments:
        k=frag[:-4]+'_'+key
        stepList=[]
        for step in upgradeScenToRun[key]:
            if 'Sim' in step:
                stepList.append(k+'_'+step)
            else:
                stepList.append(step+'_'+key)
        if numWF in numWFIB:
	    workflows[numWF] = [ upgradeDatasetFromFragment[frag], stepList]
        numWF+=1
