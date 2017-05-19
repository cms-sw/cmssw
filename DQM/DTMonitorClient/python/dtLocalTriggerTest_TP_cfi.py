import FWCore.ParameterSet.Config as cms

dtTPTriggerTest = cms.EDAnalyzer("DTLocalTriggerTPTest",
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
)

from Configuration.Eras.Modifier_run2_common_cff import run2_common
run2_common.toModify( dtTPTriggerTest, hwSources = cms.untracked.vstring('TM'))

