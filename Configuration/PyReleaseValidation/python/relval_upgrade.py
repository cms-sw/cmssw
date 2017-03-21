
# import the definition of the steps and input files:
from  Configuration.PyReleaseValidation.relval_steps import *

# here only define the workflows as a combination of the steps defined above:
workflows = Matrix()

# each workflow defines a name and a list of steps to be done. 
# if no explicit name/label given for the workflow (first arg),
# the name of step1 will be used



#just define all of them

for year in upgradeKeys:
    for i,key in enumerate(upgradeKeys[year]):
        numWF=numWFAll[year][i]
        for frag in upgradeFragments:
            k=frag[:-4]+'_'+key
            stepList=[]
            for step in upgradeProperties[year][key]['ScenToRun']:                    
                if 'Sim' in step:
                    if 'HLBeamSpotFull' in step and '14TeV' in frag:
                        step = 'GenSimHLBeamSpotFull14'
                    stepList.append(k+'_'+step)
                else:
                    stepList.append(step+'_'+key)
            workflows[numWF] = [ upgradeDatasetFromFragment[frag], stepList]

            # special workflows for tracker
            if (upgradeDatasetFromFragment[frag]=="TTbar_13" or upgradeDatasetFromFragment[frag]=="TTbar_14TeV") and not 'PU' in key:
                stepListTk=[]
                hasHarvest = False
                for step in upgradeProperties[year][key]['ScenToRun']:
                    if 'Reco' in step:
                        step = 'RecoFull_trackingOnly'
                    if 'HARVEST' in step:
                        step = 'HARVESTFull_trackingOnly'
                        hasHarvest = True

                    if 'Sim' in step:
                        stepListTk.append(k+'_'+step)
                    else:
                        stepListTk.append(step+'_'+key)
                 
                if hasHarvest:
                    workflows[numWF+0.1] = [ upgradeDatasetFromFragment[frag], stepListTk]

            numWF+=1
