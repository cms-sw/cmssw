
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

neutronKeys = ['2023D17','2023D19','2023D21','2023D22','2023D23','2023D24','2023D25','2023D28','2023D29','2023D30','2023D31','2023D33','2023D34','2023D35','2023D36','2023D37']
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
                    if stepType == 'Premix':
                        # Premixing stage1
                        #
                        # This is a hack which should be placed somewhere else, but likely requires more massive changes for "proper" PUPRMX treatment
                        #
                        # On one hand the place where upgradeWorkflowComponents.upgradeProperties[year][...PU]
                        # are defined from the noPU workflows would be a logical place. On the other hand, that
                        # would need the premixing workflows to be defined in upgradeWorkflowComponents.upgradeKeys[year]
                        # dictionary, which would further mean that we would get full set of additional workflows for
                        # premixing, while the preferred solution would be to define the premixing workflows as variations of the PU workflows.
                        s = step.replace('GenSim', 'Premix')
                        if not s in upgradeSteps[stepType]['PU']:
                            continue
                        s = s + 'PU' # later processing requires to have PU here
                        # Hardcode nu gun fragment below in order to use it for combined stage1+stage2
                        # Anyway all other fragments are irrelevant for premixing stage1
                        stepList[stepType].append(stepMaker(key,"SingleNuE10_cf",s,upgradeSteps[stepType]['suffix']))
                    elif (stepType is not 'baseline') and ( ('PU' in step and step.replace('PU','') in upgradeSteps[stepType]['PU']) or (step in upgradeSteps[stepType]['steps']) ):
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
                # skip ALCA and Nano
                trackingVariations = ['trackingOnly','trackingRun2','trackingOnlyRun2','trackingLowPU','pixelTrackingOnly','pixelTrackingOnlyRiemannFit','pixelTrackingOnlyRiemannFitGPU','pixelTrackingOnlyGPU']
                for tv in trackingVariations:
                    stepList[tv] = [s for s in stepList[tv] if (("ALCA" not in s) and ("Nano" not in s))]
                workflows[numWF+upgradeSteps['trackingOnly']['offset']] = [ upgradeDatasetFromFragment[frag], stepList['trackingOnly']]
                if '2017' in key:
                    for tv in trackingVariations[1:]:
                        workflows[numWF+upgradeSteps[tv]['offset']] = [ upgradeDatasetFromFragment[frag], stepList[tv]]
                elif '2018' in key:
                    for tv in trackingVariations:
                        if not "pixelTrackingOnly" in tv:
                            continue
                        workflows[numWF+upgradeSteps[tv]['offset']] = [ upgradeDatasetFromFragment[frag], stepList[tv]]

            # special workflows for HE
            if upgradeDatasetFromFragment[frag]=="TTbar_13" and '2018' in key:
                workflows[numWF+upgradeSteps['heCollapse']['offset']] = [ upgradeDatasetFromFragment[frag], stepList['heCollapse']]

            # premixing stage1, only for NuGun
            if upgradeDatasetFromFragment[frag]=="NuGun" and 'PU' in key and '2023' in key:
                workflows[numWF+upgradeSteps['Premix']['offset']] = [upgradeDatasetFromFragment[frag], stepList['Premix']]

            # premixing stage2, only for ttbar for time being
            if 'PU' in key and '2023' in key and upgradeDatasetFromFragment[frag]=="TTbar_14TeV":
                slist = []
                for step in stepList['baseline']:
                    s = step
                    if "Digi" in step or "Reco" in step:
                        s = s.replace("PU", "PUPRMX", 1)
                    slist.append(s)
                workflows[numWF+premixS2_offset] = [upgradeDatasetFromFragment[frag], slist]

                # Combined stage1+stage2
                workflows[numWF+premixS1S2_offset] = [upgradeDatasetFromFragment[frag], # Signal fragment
                                                      [slist[0]] +                      # Start with signal generation
                                                      stepList['Premix'] +              # Premixing stage1
                                                      [slist[1].replace("PUPRMX", "PUPRMXCombined")] + # Premixing stage2, customized for the combined (defined in relval_steps.py)
                                                      slist[2:]]                        # Remaining standard steps

            numWF+=1
