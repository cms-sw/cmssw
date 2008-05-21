#G.Benelli Feb 7 2008
#This fragment is used to have the random generator seeds saved to test
#simulation reproducibility. Anothe fragment then allows to run on the
#root output of cmsDriver.py to test reproducibility.

import FWCore.ParameterSet.Config as cms
def customise(process):
    #Renaming the process
    process.__dict__['_Process__name']='SIMDIGISavingSeeds'
    #Storing the random seeds
    process.rndmStore=cms.EDProducer("RandomEngineStateProducer")
    #print process.rndmStore.__dict__
    #Adding the RandomEngine seeds to the content
    process.out_step.outputCommands.append("keep RandomEngineStates_*_*_*")
    #For some reason cmsDriver.py complains about digitisation_step,
    #so add it by hand:
    process.digitisation_step=cms.Path(process.pdigi)
    process.rndmStore_step=cms.Path(process.rndmStore)
    
    #Modifying the schedule:
    process.schedule.append(process.simulation_step)
    process.schedule.append(process.digitisation_step)
    process.schedule.append(process.rndmStore_step)
    process.schedule.append(process.outpath)
    #Adding SimpleMemoryCheck service:
    process.SimpleMemoryCheck=cms.Service("SimpleMemoryCheck",
                                          ignoreTotal=cms.untracked.int32(1),
                                          oncePerEventMode=cms.untracked.bool(True))
    #Adding Timing service:
    process.Timing=cms.Service("Timing")
    
    #Tweak Message logger to dump G4cout and G4cerr messages in G4msg.log
    #print process.MessageLogger.__dict__
    process.MessageLogger.destinations=cms.untracked.vstring('warnings'
                                                             , 'errors'
                                                             , 'infos'
                                                             , 'debugs'
                                                             , 'cout'
                                                             , 'cerr'
                                                             , 'G4msg'
                                                             )
    process.MessageLogger.categories=cms.untracked.vstring('FwkJob'
                                                           ,'FwkReport'
                                                           ,'FwkSummary'
                                                           ,'Root_NoDictionary'
                                                           ,'G4cout'
                                                           ,'G4cerr'
                                                           )
    process.MessageLogger.cerr = cms.untracked.PSet(
        noTimeStamps = cms.untracked.bool(True)
        )
    process.MessageLogger.G4msg =  cms.untracked.PSet(
        noTimeStamps = cms.untracked.bool(True)
        ,threshold = cms.untracked.string('INFO')
        ,INFO = cms.untracked.PSet(limit = cms.untracked.int32(0))
        ,G4cout = cms.untracked.PSet(limit = cms.untracked.int32(-1))
        ,G4cerr = cms.untracked.PSet(limit = cms.untracked.int32(-1))
        )
    return(process)
