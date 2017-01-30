# import the definition of the steps and input files:
from  Configuration.PyReleaseValidation.relval_steps import *

# here only define the workflows as a combination of the steps defined above:
workflows = Matrix()

# each workflow defines a name and a list of steps to be done. 
# if no explicit name/label given for the workflow (first arg),
# the name of step1 will be used



#just define all of them

#2023 WFs to run in IB (TenMuE_0_200, TTbar, ZEE, MinBias)
numWFIB = [20021.0,20034.0,20046.0,20053.0] #2023D7 scenario
numWFIB.extend([20421.0,20434.0,20446.0,20453.0]) #2023D10
numWFIB.extend([21221.0,21234.0,21246.0,21253.0]) #2023D4
numWFIB.extend([23221.0,23234.0,23246.0,23253.0]) #2023D8
numWFIB.extend([23621.0,23634.0,23646.0,23653.0]) #2023D9
for i,key in enumerate(upgradeKeys[2023]):
    numWF=numWFAll[2023][i]
    for frag in upgradeFragments:
        k=frag[:-4]+'_'+key
        stepList=[]
        for step in upgradeProperties[2023][key]['ScenToRun']:
            if 'Sim' in step:
                if 'HLBeamSpotFull' in step and '14TeV' in frag:
                    step = 'GenSimHLBeamSpotFull14'
                stepList.append(k+'_'+step)
            else:
                stepList.append(step+'_'+key)
        if numWF in numWFIB:
	    workflows[numWF] = [ upgradeDatasetFromFragment[frag], stepList]
        numWF+=1
