import FWCore.ParameterSet.Config as cms

triggerLutTest = cms.EDFilter("DTLocalTriggerLutTest",
    # prescale factor (in luminosity blocks) to perform client analysis
    diagnosticPrescale = cms.untracked.int32(1),
    # kind of trigger data processed by DTLocalTriggerTask
    hwSources = cms.untracked.vstring('DCC', 
        'DDU'),
    # false if DTLocalTriggerTask used LTC digis
    localrun = cms.untracked.bool(True),
    # root folder for booking of histograms
    folderRoot = cms.untracked.string('')
)


