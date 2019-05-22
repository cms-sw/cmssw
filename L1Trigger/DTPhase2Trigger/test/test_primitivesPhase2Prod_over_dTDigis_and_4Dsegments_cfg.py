import FWCore.ParameterSet.Config as cms

process = cms.Process("L1DTTrigPhase2Prod")

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cff")
process.load("Geometry.DTGeometry.dtGeometry_cfi")
process.DTGeometryESModule.applyAlignment = False

process.load("L1Trigger.DTPhase2Trigger.dtTriggerPhase2PrimitiveDigis_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff")
process.GlobalTag.globaltag = "80X_dataRun2_2016SeptRepro_v7"

process.load("Phase2L1Trigger.CalibratedDigis.CalibratedDigis_cfi")
process.load("L1Trigger.DTPhase2Trigger.dtTriggerPhase2PrimitiveDigis_cfi")

process.source = cms.Source("PoolSource",fileNames = cms.untracked.vstring('file:/eos/cms/store/user/folguera/P2L1TUpgrade/digis_segments_Run2016BSingleMuonRAW-RECO_camilo.root'))
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))

process.out = cms.OutputModule("PoolOutputModule",
                               outputCommands = cms.untracked.vstring('keep *'),
                               fileName = cms.untracked.string('DTTriggerPhase2Primitives.root')
)

process.p = cms.Path(process.CalibratedDigis*process.dtTriggerPhase2PrimitiveDigis)
process.this_is_the_end = cms.EndPath(process.out)






