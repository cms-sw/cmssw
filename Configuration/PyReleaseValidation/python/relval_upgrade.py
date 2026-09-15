# import the definition of the steps and input files:
from  Configuration.PyReleaseValidation.relval_steps import *
from .MatrixUtil import Matrix

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

## ... but we don't need all the flavors for the GenOnly
def notForGenOnly(key,specialType):
    return "GenOnly" in key and specialType != 'baseline'

# the special workflows that customise a given step; this depends on the step name only,
# and the step names come from a small fixed set, so answer it once per step name
customisersByStep = {}
def customisersFor(step):
    customisers = customisersByStep.get(step)
    if customisers is None:
        isPU = 'PU' in step
        stepNoPU = step.replace('PU','') if isPU else None
        customisers = [(specialType,specialWF) for specialType,specialWF in upgradeWFs.items()
                       if specialType != 'baseline' and ((isPU and stepNoPU in specialWF.PU) or (step in specialWF.steps))]
        customisersByStep[step] = customisers
    return customisers

for year in upgradeKeys:
    for i,key in enumerate(upgradeKeys[year]):
        numWF=numWFAll[year][i]
        # neither the applicable flavors nor the presence of a harvesting step depend on
        # the fragment, so resolve them once per key
        activeWFs = [(specialType,specialWF) for specialType,specialWF in upgradeWFs.items()
                     if not notForGenOnly(key,specialType)]
        scenToRun = upgradeProperties[year][key]['ScenToRun']
        hasHarvest = any('HARVEST' in step for step in scenToRun)
        # the resolved step names and the steps each flavor customises depend on the
        # fragment only through the flags below, so resolve them once per combination
        byFragmentClass={}
        for frag,info in upgradeFragments.items():
            # phase2-specific fragments are skipped in phase1
            if ("CE_E" in frag or "CE_H" in frag) and year==2017:
                numWF += 1
                continue
            fragName = frag[:-4]
            is14TeV = '14TeV' in frag
            isCloseBy = 'CloseBy' in frag or 'CE_E' in frag or 'CE_H' in frag
            isDisplaced = 'DisplacedParticleGun' in frag
            if (is14TeV,isCloseBy,isDisplaced) in byFragmentClass:
                resolved,customised = byFragmentClass[(is14TeV,isCloseBy,isDisplaced)]
            else:
                resolved=[]
                for step in scenToRun:
                    stepMaker = makeStepName
                    if 'Sim' in step and 'Fast' not in step and step != "Sim":
                        if isDisplaced:
                            step = 'GenSimDisplaced'
                        elif 'HLBeamSpot' in step:
                            if is14TeV:
                                step = 'GenSimHLBeamSpot14'
                            elif isCloseBy:
                                step = 'GenSimHLBeamSpotCloseBy'
                        elif isCloseBy:
                            step = 'GenSimCloseBy'
                        stepMaker = makeStepNameSim
                    elif 'Gen' in step:
                        if 'HLBeamSpot' in step:
                            if is14TeV:
                                step = 'GenHLBeamSpot14'
                        stepMaker = makeStepNameSim
                    resolved.append((stepMaker,step))
                # the steps each flavor customises: a flavor that customises none of them
                # ends up with the baseline step list, which workflow_() drops as spurious
                customised={}
                for index,(stepMaker,step) in enumerate(resolved):
                    for specialType,specialWF in customisersFor(step):
                        customised.setdefault(specialType,[]).append(index)
                byFragmentClass[(is14TeV,isCloseBy,isDisplaced)] = (resolved,customised)

            baseStepList = [stepMaker(key,fragName,step,'') for stepMaker,step in resolved]

            for specialType,specialWF in activeWFs:
                accepted = None
                if specialType=='baseline':
                    stepList = list(baseStepList)
                else:
                    modified = customised.get(specialType)
                    # PMXS1 truncates its list below, so it differs from the baseline one
                    # even when it customises no step
                    if modified is None and specialType!="PMXS1":
                        continue
                    # a flavor whose condition() rejects this workflow contributes
                    # nothing, so its step list is never needed
                    if not specialWF.conditionUsesStepList:
                        accepted = specialWF.condition(info.dataset, None, key, hasHarvest)
                        if not accepted:
                            continue
                    if modified is None: modified = []
                    stepList = []
                    for index,(stepMaker,step) in enumerate(resolved):
                        if index not in modified:
                            stepList.append(baseStepList[index])
                            continue
                        stepList.append(stepMaker(key,fragName,step,specialWF.suffix))
                        # hack to add an extra step
                        if 'ProdLike' in specialType:
                            if 'Reco' in step: # handles both Reco, RecoFakeHLT and RecoGlobal
                                stepWoFakeHLT = step.replace('FakeHLT','') # ignore "FakeHLT" from step
                                stepList.append(stepMaker(key,fragName,stepWoFakeHLT.replace('RecoGlobal','MiniAOD').replace('RecoNano','MiniAOD').replace('Reco','MiniAOD'),specialWF.suffix))
                                if 'RecoNano' in stepWoFakeHLT:
                                    stepList.append(stepMaker(key,fragName,stepWoFakeHLT.replace('RecoNano','Nano'),specialWF.suffix))
                        # hack to add extra HLT75e33 step for Phase-2
                        if 'HLT75e33' in specialType:
                            if 'RecoGlobal' in step:
                                stepList.append(stepMaker(key,fragName,step.replace('RecoGlobal','HLT75e33'),specialWF.suffix))
                        # similar hacks for premixing
                        if 'PMX' in specialType:
                            if 'GenSim' in step or 'Gen' in step:
                                s = step.replace('GenSim','Premix').replace('Gen','Premix')+'PU' # later processing requires to have PU here
                                if step in specialWF.PU:
                                    stepMade = stepMaker(key,'PREMIX',s,specialWF.suffix)
                                    # append for combined
                                    if 'S2' in specialType: stepList.append(stepMade)
                                    # replace for s1
                                    else: stepList[-1] = stepMade
                        # similar hack for fastpu
                        if 'HybridPU' in specialType:
                            if 'GenSim' in step:
                                s = step.replace('GenSim','GenSimFS')+'PU' # later processing requires to have PU here
                                if step in specialWF.PU:
                                    stepMade = stepMaker(key,'HYBRID',s,specialWF.suffix)
                                    # append for combined
                                    if 'S2' in specialType: stepList.append(stepMade)
                    # remove other steps for premixS1
                    if specialType=="PMXS1":
                        stepList = stepList[:1]
                if accepted:
                    specialWF.workflow_(workflows, numWF, info.dataset, stepList, key)
                else:
                    specialWF.workflow(workflows, numWF, info.dataset, stepList, key, hasHarvest)
            numWF+=1
