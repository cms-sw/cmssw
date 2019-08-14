
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

neutronKeys = [x for x in upgradeKeys[2026] if 'PU' not in x]
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
                        # hack to add an extra step
                        if stepType == 'ProdLike' and 'RecoFullGlobal' in step:
                            stepList[stepType].append(stepMaker(key,frag[:-4],step.replace('RecoFullGlobal','MiniAODFullGlobal'),upgradeSteps[stepType]['suffix']))
                    else:
                        stepList[stepType].append(stepMaker(key,frag[:-4],step,upgradeSteps['baseline']['suffix']))

            workflows[numWF] = [ upgradeDatasetFromFragment[frag], stepList['baseline']]

            # special workflows for neutron bkg sim
            if any(upgradeDatasetFromFragment[frag]==nfrag for nfrag in neutronFrags) and any(nkey in key for nkey in neutronKeys):
                workflows[numWF+upgradeSteps['Neutron']['offset']] = [ upgradeDatasetFromFragment[frag]+"_Neutron", stepList['Neutron']]

            # special workflows for tracker
            if (upgradeDatasetFromFragment[frag]=="TTbar_13" or upgradeDatasetFromFragment[frag]=="TTbar_14TeV") and not 'PU' in key and hasHarvest:
                # skip ALCA and Nano
                trackingVariations = ['trackingOnly','trackingRun2','trackingOnlyRun2','trackingLowPU','pixelTrackingOnly']
                for tv in trackingVariations:
                    stepList[tv] = [s for s in stepList[tv] if (("ALCA" not in s) and ("Nano" not in s))]
                workflows[numWF+upgradeSteps['trackingOnly']['offset']] = [ upgradeDatasetFromFragment[frag], stepList['trackingOnly']]
                if '2017' in key:
                    for tv in trackingVariations[1:]:
                        workflows[numWF+upgradeSteps[tv]['offset']] = [ upgradeDatasetFromFragment[frag], stepList[tv]]
                elif '2018' in key:
                    workflows[numWF+upgradeSteps['pixelTrackingOnly']['offset']] = [ upgradeDatasetFromFragment[frag], stepList['pixelTrackingOnly']]

            # special workflows for HGCAL/TICL
            if (upgradeDatasetFromFragment[frag]=="CloseByParticleGun") and ('2026' in key):
                TICLVariations = ['TICLOnly', 'TICLFullReco']
                # Skip Hharvesting for TICLOnly
                for tv in TICLVariations:
                    if 'TICLOnly' in tv:
                        stepList[tv] = [s for s in stepList[tv] if ("HARVEST" not in s)]
                for tv in TICLVariations:
                    workflows[numWF+upgradeSteps[tv]['offset']] = [ upgradeDatasetFromFragment[frag], stepList[tv]]

            # special workflows for HE
            if upgradeDatasetFromFragment[frag]=="TTbar_13" and '2018' in key:
                workflows[numWF+upgradeSteps['heCollapse']['offset']] = [ upgradeDatasetFromFragment[frag], stepList['heCollapse']]

            # workflow for profiling
            if upgradeDatasetFromFragment[frag]=="TTbar_14TeV" and '2026' in key:
                workflows[numWF+upgradeSteps['ProdLike']['offset']] = [ upgradeDatasetFromFragment[frag]+"_ProdLike", stepList['ProdLike']]

            # special workflows for ParkingBPH
            if upgradeDatasetFromFragment[frag]=="TTbar_13" and '2018' in key:
                workflows[numWF+upgradeSteps['ParkingBPH']['offset']] = [ upgradeDatasetFromFragment[frag], stepList['ParkingBPH']]

            inclPremix = 'PU' in key
            if inclPremix:
                inclPremix = False
                for y in ['2021', '2023', '2024', '2026']:
                    if y in key:
                        inclPremix = True
                        continue

            # premixing stage1, only for NuGun
            if inclPremix and upgradeDatasetFromFragment[frag]=="NuGun":
                workflows[numWF+upgradeSteps['Premix']['offset']] = [upgradeDatasetFromFragment[frag], stepList['Premix']]

            # premixing stage2, only for ttbar for time being
            if inclPremix and upgradeDatasetFromFragment[frag]=="TTbar_14TeV":
                slist = []
                for step in stepList['baseline']:
                    s = step
                    if "Digi" in step or "Reco" in step:
                        s = s.replace("PU", "PUPRMX", 1)
                    slist.append(s)
                workflows[numWF+premixS2_offset] = [upgradeDatasetFromFragment[frag], slist]

                # Combined stage1+stage2
                def nano(s):
                    if "Nano" in s:
                        if "_" in s:
                            return s.replace("_", "PUPRMXCombined_")
                        return s+"PUPRMXCombined"
                    return s
                workflows[numWF+premixS1S2_offset] = [upgradeDatasetFromFragment[frag], # Signal fragment
                                                      [slist[0]] +                      # Start with signal generation
                                                      stepList['Premix'] +              # Premixing stage1
                                                      [slist[1].replace("PUPRMX", "PUPRMXCombined")] + # Premixing stage2, customized for the combined (defined in relval_steps.py)
                                                      map(nano, slist[2:])]             # Remaining standard steps

            numWF+=1
