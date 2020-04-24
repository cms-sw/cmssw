import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

triggerTest = DQMEDHarvester("DTLocalTriggerTest",
    # prescale factor (in luminosity blocks) to perform client analysis
    diagnosticPrescale = cms.untracked.int32(1),
    # run in online environment
    runOnline = cms.untracked.bool(True),
    # kind of trigger data processed by DTLocalTriggerTask
    hwSources = cms.untracked.vstring('TM','DDU'),
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
    # DDU-TM matching tests tresholds
    matchingFracError     = cms.untracked.double(0.65),
    matchingFracWarning   = cms.untracked.double(0.85),
    nEventsCert = cms.untracked.int32(1000)
                             

)

from Configuration.Eras.Modifier_run2_common_cff import run2_common
run2_common.toModify( triggerTest, hwSources = cms.untracked.vstring('TM'))

