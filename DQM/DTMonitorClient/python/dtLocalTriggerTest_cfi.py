import FWCore.ParameterSet.Config as cms

triggerTest = cms.EDAnalyzer("DTLocalTriggerTest",
    # prescale factor (in luminosity blocks) to perform client analysis
    diagnosticPrescale = cms.untracked.int32(1),
    # run in online environment
    runOnline = cms.untracked.bool(True),
    # kind of trigger data processed by DTLocalTriggerTask
    hwSources = cms.untracked.vstring('DCC','DDU'),
    # false if DTLocalTriggerTask used LTC digis
    localrun = cms.untracked.bool(True),                         
    # root folder for booking of histograms
    folderRoot = cms.untracked.string(''),
    # correlated fraction amd 2nd fraction tests tresholds
    corrFracError     = cms.untracked.double(0.50),
    corrFracWarning   = cms.untracked.double(0.60),
    secondFracError   = cms.untracked.double(0.20),
    secondFracWarning = cms.untracked.double(0.10)
)


