import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

triggerLutTest = DQMEDHarvester("DTTriggerLutTest",
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
    validRange = cms.untracked.double(2.),
    # thershold for warning & errors in phi/phib tests
    thresholdWarnPhi = cms.untracked.double(.95),
    thresholdErrPhi  = cms.untracked.double(.90),
    thresholdWarnPhiB = cms.untracked.double(.95),
    thresholdErrPhiB  = cms.untracked.double(.90),
    # detailed analysis flag
    detailedAnalysis = cms.untracked.bool(False),
    # enable/ disable dynamic booking
    staticBooking = cms.untracked.bool(True)                                   
)


