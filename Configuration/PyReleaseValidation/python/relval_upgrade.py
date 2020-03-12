import six
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

#just define all of them

for year in upgradeKeys:
    for i,key in enumerate(upgradeKeys[year]):
        numWF=numWFAll[year][i]
        for frag,info in six.iteritems(upgradeFragments):
            # phase2-specific fragments are skipped in phase1
            if ("CE_E" in frag or "CE_H" in frag) and year==2017:
                numWF += 1
                continue
            stepList={}
            for specialType in upgradeWFs.keys():
                stepList[specialType] = []
            hasHarvest = False
            for step in upgradeProperties[year][key]['ScenToRun']:                    
                stepMaker = makeStepName
                if 'Sim' in step:
                    if 'HLBeamSpotFull' in step and '14TeV' in frag:
                        step = 'GenSimHLBeamSpotFull14'
                    stepMaker = makeStepNameSim
                
                if 'HARVEST' in step: hasHarvest = True

                for specialType,specialWF in six.iteritems(upgradeWFs):
                    # use variation only when available
                    if specialType == 'Premix':
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
                        if not s in specialWF.PU:
                            continue
                        s = s + 'PU' # later processing requires to have PU here
                        # Hardcode nu gun fragment below in order to use it for combined stage1+stage2
                        # Anyway all other fragments are irrelevant for premixing stage1
                        stepList[specialType].append(stepMaker(key,'PREMIX',s,specialWF.suffix))
                    elif (specialType is not 'baseline') and ( ('PU' in step and step.replace('PU','') in specialWF.PU) or (step in specialWF.steps) ):
                        stepList[specialType].append(stepMaker(key,frag[:-4],step,specialWF.suffix))
                        # hack to add an extra step
                        if specialType == 'ProdLike' and 'RecoFullGlobal' in step:
                            stepList[specialType].append(stepMaker(key,frag[:-4],step.replace('RecoFullGlobal','MiniAODFullGlobal'),specialWF.suffix))
                        elif specialType == 'ProdLike' and 'RecoFull' in step:
                            stepList[specialType].append(stepMaker(key,frag[:-4],step.replace('RecoFull','MiniAODFullGlobal'),specialWF.suffix))
                    else:
                        stepList[specialType].append(stepMaker(key,frag[:-4],step,''))

            for specialType,specialWF in six.iteritems(upgradeWFs):
                specialWF.workflow(workflows, numWF, info.dataset, stepList[specialType], key, hasHarvest)

            inclPremix = 'PU' in key
            if inclPremix:
                inclPremix = False
                for y in ['2021', '2023', '2024', '2026']:
                    if y in key:
                        inclPremix = True
                        continue
            if inclPremix:
                # premixing stage1, only for NuGun
                if info.dataset=="NuGun":
                    # The first element of the list sets the dataset name(?)
                    datasetName = 'PREMIXUP' + key[2:].replace("PU", "").replace("Design", "") + '_PU25'
                    workflows[numWF+upgradeWFs['Premix'].offset] = [datasetName, stepList['Premix']]

                # premixing stage2
                slist = []
                for step in stepList['baseline']:
                    s = step
                    if "Digi" in step or "Reco" in step:
                        s = s.replace("PU", "PUPRMX", 1)
                    slist.append(s)
                workflows[numWF+upgradeWFs['premixS2'].offset] = [info.dataset, slist]

                # Combined stage1+stage2
                def nano(s):
                    if "Nano" in s:
                        if "_" in s:
                            return s.replace("_", "PUPRMXCombined_")
                        return s+"PUPRMXCombined"
                    return s
                workflows[numWF+upgradeWFs['premixS1S2'].offset] = [info.dataset, # Signal fragment
                                                      [slist[0]] +                      # Start with signal generation
                                                      stepList['Premix'] +              # Premixing stage1
                                                      [slist[1].replace("PUPRMX", "PUPRMXCombined")] + # Premixing stage2, customized for the combined (defined in relval_steps.py)
                                                      list(map(nano, slist[2:]))]             # Remaining standard steps

            numWF+=1
