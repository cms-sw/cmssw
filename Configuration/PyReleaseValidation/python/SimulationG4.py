#G.Benelli Dec 21 2007
#This fragment is used for the simulation (SIM) step
#It includes a MessageLogger tweak to dump G4msg.log
#in addition to the the SimpleMemoryCheck and Timing
#services output for the log of the simulation
#performance candles.
#It is meant to be used with the cmsDriver.py option
#--customise in the following fashion:
#E.g.
#./cmsDriver.py HZZLLLL -e 190 -n 50 --step=SIM --customise=SimulationG4.py >& HZZLLLL_190_GEN.log&
#or
#./cmsDriver.py MINBIAS -n 50 --step=SIM --customise=SimulationG4.py >& MINBIAS_GEN.log&


import FWCore.ParameterSet.Config as cms
def customise(process):
    #Adding SimpleMemoryCheck service:
    process.SimpleMemoryCheck=cms.Service("SimpleMemoryCheck",
                                          ignoreTotal=cms.untracked.int32(1),
                                          oncePerEventMode=cms.untracked.bool(True))
    #Adding Timing service:
    process.Timing=cms.Service("Timing")
    
    #Tweak Message logger to dump G4cout and G4cerr messages in G4msg.log
    #print process.MessageLogger.__dict__
    process.MessageLogger.destinations=cms.untracked.vstring('cout'
                                                             ,'cerr'
                                                             ,'G4msg'
                                                             )
    process.MessageLogger.categories=cms.untracked.vstring('FwkJob'
                                                           ,'FwkReport'
                                                           ,'FwkSummary'
                                                           ,'Root_NoDictionary'
                                                           ,'TimeReport'
                                                           ,'TimeModule'
                                                           ,'TimeEvent'
                                                           ,'MemoryCheck'
                                                           ,'PhysicsList'
                                                           ,'G4cout'
                                                           ,'G4cerr'
                                                           )
    #Configuring the G4msg.log output
    process.MessageLogger.G4msg =  cms.untracked.PSet(
        noTimeStamps = cms.untracked.bool(True)
        #First eliminate unneeded output
        ,threshold = cms.untracked.string('INFO')
        ,INFO = cms.untracked.PSet(limit = cms.untracked.int32(0))
        ,FwkReport = cms.untracked.PSet(limit = cms.untracked.int32(0))
        ,FwkSummary = cms.untracked.PSet(limit = cms.untracked.int32(0))
        ,Root_NoDictionary = cms.untracked.PSet(limit = cms.untracked.int32(0))
        ,FwkJob = cms.untracked.PSet(limit = cms.untracked.int32(0))
        ,TimeReport = cms.untracked.PSet(limit = cms.untracked.int32(0))
        ,TimeModule = cms.untracked.PSet(limit = cms.untracked.int32(0))
        ,TimeEvent = cms.untracked.PSet(limit = cms.untracked.int32(0))
        ,MemoryCheck = cms.untracked.PSet(limit = cms.untracked.int32(0))
        #TimeModule, TimeEvent, TimeReport are written to LogAsbolute instead of LogInfo with a category
        #so they cannot be eliminated from any destination (!) unless one uses the summaryOnly option
        #in the Timing Service... at the price of silencing the output needed for the TimingReport profiling
        #
        #Then add the wanted ones:
        ,PhysicsList = cms.untracked.PSet(limit = cms.untracked.int32(-1))
        ,G4cout = cms.untracked.PSet(limit = cms.untracked.int32(-1))
        ,G4cerr = cms.untracked.PSet(limit = cms.untracked.int32(-1))
        )
    return(process)
