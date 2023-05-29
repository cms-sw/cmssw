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
        for frag,info in upgradeFragments.items():
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
                if 'Sim' in step and 'Fast' not in step:
                    if 'HLBeamSpot' in step:
                        if '14TeV' in frag:
                            step = 'GenSimHLBeamSpot14'
                        if 'CloseByParticle' in frag or 'CE_E' in frag or 'CE_H' in frag:
                            step = 'GenSimHLBeamSpotHGCALCloseBy'
                    stepMaker = makeStepNameSim
                elif 'Gen' in step:
                    stepMaker = makeStepNameSim
                
                if 'HARVEST' in step: hasHarvest = True

                for specialType,specialWF in upgradeWFs.items():
                    if (specialType != 'baseline') and ( ('PU' in step and step.replace('PU','') in specialWF.PU) or (step in specialWF.steps) ):
                        stepList[specialType].append(stepMaker(key,frag[:-4],step,specialWF.suffix))
                        # hack to add an extra step
                        if 'ProdLike' in specialType:
                            if 'Reco' in step: # handles both Reco, RecoFakeHLT and RecoGlobal
                                stepWoFakeHLT = step.replace('FakeHLT','') # ignore "FakeHLT" from step
                                stepList[specialType].append(stepMaker(key,frag[:-4],stepWoFakeHLT.replace('RecoGlobal','MiniAOD').replace('RecoNano','MiniAOD').replace('Reco','MiniAOD'),specialWF.suffix))
                                if 'RecoNano' in stepWoFakeHLT:
                                    stepList[specialType].append(stepMaker(key,frag[:-4],stepWoFakeHLT.replace('RecoNano','Nano'),specialWF.suffix))
                        # hack to add extra HLT75e33 step for Phase-2
                        if 'HLT75e33' in specialType:
                            if 'RecoGlobal' in step:
                                stepList[specialType].append(stepMaker(key,frag[:-4],step.replace('RecoGlobal','HLT75e33'),specialWF.suffix))
                        # similar hacks for premixing
                        if 'PMX' in specialType:
                            if 'GenSim' in step:
                                s = step.replace('GenSim','Premix')+'PU' # later processing requires to have PU here
                                if step in specialWF.PU:
                                    stepMade = stepMaker(key,'PREMIX',s,specialWF.suffix)
                                    # append for combined
                                    if 'S2' in specialType: stepList[specialType].append(stepMade)
                                    # replace for s1
                                    else: stepList[specialType][-1] = stepMade
                    else:
                        stepList[specialType].append(stepMaker(key,frag[:-4],step,''))

            for specialType,specialWF in upgradeWFs.items():
                # remove other steps for premixS1
                if specialType=="PMXS1":
                    stepList[specialType] = stepList[specialType][:1]
                specialWF.workflow(workflows, numWF, info.dataset, stepList[specialType], key, hasHarvest)

            numWF+=1
