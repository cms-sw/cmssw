# import the definition of the steps and input files:
from  Configuration.PyReleaseValidation.relval_steps import *

# here only define the workflows as a combination of the steps defined above:
workflows = Matrix()

# each workflow defines a name and a list of steps to be done. 
# if no explicit name/label given for the workflow (first arg),
# the name of step1 will be used



#just define all of them

numWFStart=20000
numWFSkip=200
#2023 WFs to run in IB (TenMuE_0_200, TTbar, ZEE, MinBias)
numWFIB = [20021.0,20024.0,20025.0,20026.0] #2023D1 scenario
numWFIB.extend([20421.0,20424.0,20425.0,20426.0]) #2023D2
numWFIB.extend([20821.0,20824.0,20825.0,20826.0]) #2023D3
numWFIB.extend([21221.0,21224.0,21225.0,21226.0]) #2023D4
numWFIB.extend([22424.0]) #2023D3Timing
for i,key in enumerate(upgradeKeys[2023]):
    numWF=numWFStart+i*numWFSkip
    for frag in upgradeFragments:
        k=frag[:-4]+'_'+key
        stepList=[]
        for step in upgradeProperties[2023][key]['ScenToRun']:
            if 'Sim' in step:
                stepList.append(k+'_'+step)
            else:
                stepList.append(step+'_'+key)
        if numWF in numWFIB:
	    workflows[numWF] = [ upgradeDatasetFromFragment[frag], stepList]
        numWF+=1
