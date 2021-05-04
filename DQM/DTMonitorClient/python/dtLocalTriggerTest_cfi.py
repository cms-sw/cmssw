import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

triggerTest = DQMEDHarvester("DTLocalTriggerTest",
    # prescale factor (in luminosity blocks) to perform client analysis
    diagnosticPrescale = cms.untracked.int32(1),
    # run in online environment
    runOnline = cms.untracked.bool(True),
    # kind of trigger data processed by DTLocalTriggerTask
    hwSources = cms.untracked.vstring('TM'),
    # false if DTLocalTriggerTask used LTC digis
    localrun = cms.untracked.bool(True),                         
    # root folder for booking of histograms
    folderRoot = cms.untracked.string(''),
    # correlated fraction test tresholds
    corrFracError     = cms.untracked.double(0.35),
    corrFracWarning   = cms.untracked.double(0.45),
    # second fraction test tresholds
    secondFracError   = cms.untracked.double(0.95),
    secondFracWarning = cms.untracked.double(0.8),
    nEventsCert = cms.untracked.int32(1000)
                             

)


