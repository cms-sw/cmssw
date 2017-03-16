import FWCore.ParameterSet.Config as cms

triggerLutTest = cms.EDAnalyzer("DTLocalTriggerLutTest",
    # prescale factor (in luminosity blocks) to perform client analysis
    diagnosticPrescale = cms.untracked.int32(1),
    # run in online environment
    runOnline = cms.untracked.bool(True),
    # kind of trigger data processed by DTLocalTriggerTask
    hwSources = cms.untracked.vstring('TM','DDU'),
    # false if DTLocalTriggerTask used LTC digis
    localrun = cms.untracked.bool(True),
    # enable/disable correlation plot tests
    doCorrelationStudy = cms.untracked.bool(False),
    # root folder for booking of histograms
    folderRoot = cms.untracked.string('')
)

from Configuration.Eras.Modifier_run2_common_cff import run2_common
run2_common.toModify( triggerLutTest, hwSources = cms.untracked.vstring('TM'))

from Configuration.Eras.Modifier_run2_25ns_specific_cff import run2_25ns_specific
run2_25ns_specific.toModify( triggerLutTest, hwSources = cms.untracked.vstring('TM'))

from Configuration.Eras.Modifier_run2_HI_specific_cff import run2_HI_specific
run2_HI_specific.toModify( triggerLutTest, hwSources = cms.untracked.vstring('TM'))

from Configuration.Eras.Modifier_pA_2016_cff import pA_2016
pA_2016.toModify( triggerLutTest, hwSources = cms.untracked.vstring('TM'))
