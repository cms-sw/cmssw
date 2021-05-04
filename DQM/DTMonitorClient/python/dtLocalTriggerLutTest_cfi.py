import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

triggerLutTest = DQMEDHarvester("DTLocalTriggerLutTest",
    # prescale factor (in luminosity blocks) to perform client analysis
    diagnosticPrescale = cms.untracked.int32(1),
    # run in online environment
    runOnline = cms.untracked.bool(True),
    # kind of trigger data processed by DTLocalTriggerTask
    hwSources = cms.untracked.vstring('TM'),
    # false if DTLocalTriggerTask used LTC digis
    localrun = cms.untracked.bool(True),
    # enable/disable correlation plot tests
    doCorrelationStudy = cms.untracked.bool(False),
    # root folder for booking of histograms
    folderRoot = cms.untracked.string('')
)

