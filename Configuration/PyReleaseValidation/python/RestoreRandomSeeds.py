#G.Benelli Feb 7 2008
#This fragment is used to have the random generator seeds saved to test
#simulation reproducibility. Anothe fragment then allows to run on the
#root output of cmsDriver.py to test reproducibility.

import FWCore.ParameterSet.Config as cms
def customise(process):
    #Renaming the process
    process.__dict__['_Process__name']='SIMDIGIRestoringSeeds'
    #Skipping the first 3 events:
    process.PoolSource.skipEvents=cms.untracked.uint32(3)
    #Adding RandomNumberGeneratorService
    process.RandomNumberGeneratorService.restoreStateLabel=cms.untracked.string('rndmStore')
    process.RandomNumberGeneratorService.moduleSeeds = cms.PSet(
        VtxSmeared = cms.untracked.uint32(1)
        ,g4SimHits = cms.untracked.uint32(1)
        ,mix = cms.untracked.uint32(1)
        ,siPixelDigis = cms.untracked.uint32(1)
        ,siStripDigis = cms.untracked.uint32(1)
        ,ecalUnsuppressedDigis = cms.untracked.uint32(1)
        ,hcalUnsuppressedDigis = cms.untracked.uint32(1)
        ,muonCSCDigis = cms.untracked.uint32(1)
        ,muonDTDigis = cms.untracked.uint32(1)
        ,muonRPCDigis = cms.untracked.uint32(1)
        )
    #This line is necessary to eliminate the sourceSeed line in the python configuration!
    del process.RandomNumberGeneratorService.sourceSeed
    #Adding the RandomEngine seeds to the content
    process.out_step.outputCommands.append("drop *_*_*_Sim")
    process.out_step.outputCommands.append("keep RandomEngineStates_*_*_*")
    #For some reason cmsDriver.py complains about digitisation_step,
    #so add it by hand:
    process.digitisation_step=cms.Path(process.pdigi)
    process.g4SimHits_step=cms.Path(process.g4SimHits)
    #Modifying the schedule:
    process.schedule.append(process.g4SimHits_step)
    process.schedule.append(process.digitisation_step)
    process.schedule.append(process.outpath)
    #Adding SimpleMemoryCheck service:
    process.SimpleMemoryCheck=cms.Service("SimpleMemoryCheck",
                                          ignoreTotal=cms.untracked.int32(1),
                                          oncePerEventMode=cms.untracked.bool(True))
    #Adding Timing service:
    process.Timing=cms.Service("Timing")
    return(process)
