
# import the definition of the steps and input files:
from  Configuration.PyReleaseValidation.relval_steps import *

# here only define the workflows as a combination of the steps defined above:
workflows = Matrix()

# each workflow defines a name and a list of steps to be done. 
# if no explicit name/label given for the workflow (first arg),
# the name of step1 will be used

def makeStepNameSim(key,frag,step,suffix):
    return frag+'_'+key+'_'+step+suffix

def makeStepName(key,frag,step,suffix):
   return step+suffix+'_'+key

neutronKeys = ['2023D17','2023D19']
neutronFrags = ['ZMM_14','MinBias_14TeV']

#just define all of them

for year in upgradeKeys:
    for i,key in enumerate(upgradeKeys[year]):
        numWF=numWFAll[year][i]
        for frag in upgradeFragments:
            stepList={}
            for stepType in upgradeSteps.keys():
                stepList[stepType] = []
            hasHarvest = False
            for step in upgradeProperties[year][key]['ScenToRun']:                    
                stepMaker = makeStepName
                if 'Sim' in step:
                    if 'HLBeamSpotFull' in step and '14TeV' in frag:
                        step = 'GenSimHLBeamSpotFull14'
                    stepMaker = makeStepNameSim
                
                if 'HARVEST' in step: hasHarvest = True

                for stepType in upgradeSteps.keys():
                    # use variation only when available
                    if (stepType is not 'baseline') and ( ('PU' in step and step.replace('PU','') in upgradeSteps[stepType]['PU']) or (step in upgradeSteps[stepType]['steps']) ):
                        stepList[stepType].append(stepMaker(key,frag[:-4],step,upgradeSteps[stepType]['suffix']))
                    else:
                        stepList[stepType].append(stepMaker(key,frag[:-4],step,upgradeSteps['baseline']['suffix']))

            workflows[numWF] = [ upgradeDatasetFromFragment[frag], stepList['baseline']]

            # only keep some special workflows for timing
            if upgradeDatasetFromFragment[frag]=="TTbar_14TeV" and '2023' in key:
                workflows[numWF+upgradeSteps['Timing']['offset']] = [ upgradeDatasetFromFragment[frag]+"_Timing", stepList['Timing']]

            # special workflows for neutron bkg sim
            if any(upgradeDatasetFromFragment[frag]==nfrag for nfrag in neutronFrags) and any(nkey in key for nkey in neutronKeys):
                workflows[numWF+upgradeSteps['Neutron']['offset']] = [ upgradeDatasetFromFragment[frag]+"_Neutron", stepList['Neutron']]

            # special workflows for tracker
            if (upgradeDatasetFromFragment[frag]=="TTbar_13" or upgradeDatasetFromFragment[frag]=="TTbar_14TeV") and not 'PU' in key and hasHarvest:
                workflows[numWF+upgradeSteps['trackingOnly']['offset']] = [ upgradeDatasetFromFragment[frag], stepList['trackingOnly']]

            numWF+=1
