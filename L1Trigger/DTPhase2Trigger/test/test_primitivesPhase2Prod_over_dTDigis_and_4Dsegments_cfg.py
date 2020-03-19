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

#process.source = cms.Source("PoolSource",fileNames = cms.untracked.vstring('file:/eos/cms/store/user/folguera/P2L1TUpgrade/digis_segments_Run2016BSingleMuonRAW-RECO_camilo.root'))
process.source = cms.Source("PoolSource",fileNames = cms.untracked.vstring('file:/eos/cms/store/user/folguera/P2L1TUpgrade/digis_segments_Run2016BSingleMuonRAW-RECO_camilo.root'),skipEvents=cms.untracked.uint32(1))
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))
process.dtTriggerPhase2PrimitiveDigis.dump = False
process.dtTriggerPhase2PrimitiveDigis.debug = True
process.dtTriggerPhase2PrimitiveDigis.chi2Th = cms.untracked.double(0.16)

#scenario
process.dtTriggerPhase2PrimitiveDigis.scenario = 1
process.CalibratedDigis.scenario = 1


#python tecno30
process.dtTriggerPhase2PrimitiveDigis.printPython = True 
process.dtTriggerPhase2PrimitiveDigis.printHits = False 

####################### SliceTest specials ##############################

#Chi2 -> Changing a lot lately
process.dtTriggerPhase2PrimitiveDigis.chi2Th = cms.untracked.double(0.16)

#LSB -> Position 0.025 cm instead of 0.004 cm
process.dtTriggerPhase2PrimitiveDigis.use_LSB = True
process.dtTriggerPhase2PrimitiveDigis.x_precision = cms.untracked.double(1./(10.*16.))
#process.dtTriggerPhase2PrimitiveDigis.x_precision = cms.untracked.double(0.025)
process.dtTriggerPhase2PrimitiveDigis.tanPsi_precision = cms.untracked.double(1./4096.)

#Correlate with BX
process.dtTriggerPhase2PrimitiveDigis.useBX_correlation = True
process.dtTriggerPhase2PrimitiveDigis.dBX_correlate_TP = 1

#Correlate with tanPsi
process.dtTriggerPhase2PrimitiveDigis.dTanPsi_correlate_TP = cms.untracked.double(9999./4096.)
#process.dtTriggerPhase2PrimitiveDigis.dTanPsi_correlate_TP = cms.untracked.double(900./4096.)

#Confirmation forbidden
process.dtTriggerPhase2PrimitiveDigis.allow_confirmation = False

#TanPsi stuff
process.dtTriggerPhase2PrimitiveDigis.tanPhiTh = cms.untracked.double(1.)


process.out = cms.OutputModule("PoolOutputModule",
                               outputCommands = cms.untracked.vstring('keep *'),
                               fileName = cms.untracked.string('DTTriggerPhase2Primitives.root')
)

process.p = cms.Path(process.CalibratedDigis*process.dtTriggerPhase2PrimitiveDigis)
process.this_is_the_end = cms.EndPath(process.out)






