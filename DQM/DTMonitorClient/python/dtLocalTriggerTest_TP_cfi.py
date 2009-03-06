import FWCore.ParameterSet.Config as cms

dtTPTriggerTest = cms.EDAnalyzer("DTLocalTriggerTPTest",
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
)


