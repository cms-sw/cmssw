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
#2023 WFs to run in IB (TenMuE_0_200, TTbar, ZEE, MinBias)
numWFIB = [10421.0,10424.0,10425.0,10426.0] #2023D1 scenario
numWFIB.extend([10821.0,10824.0,10825.0,10826.0]) #2023D2
numWFIB.extend([11221.0,11224.0,11225.0,11226.0]) #2023D3
numWFIB.extend([11621.0,11624.0,11625.0,11626.0]) #2023D4
for i,key in enumerate(upgradeKeys):
    numWF=numWFStart+i*numWFSkip
    for frag in upgradeFragments:
        k=frag[:-4]+'_'+key
        stepList=[]
        for step in upgradeProperties[key]['ScenToRun']:
            if 'Sim' in step:
                stepList.append(k+'_'+step)
            else:
                stepList.append(step+'_'+key)
        if numWF in numWFIB:
	    workflows[numWF] = [ upgradeDatasetFromFragment[frag], stepList]
        numWF+=1
