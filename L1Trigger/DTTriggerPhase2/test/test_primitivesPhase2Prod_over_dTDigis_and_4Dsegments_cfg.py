import FWCore.ParameterSet.Config as cms

process = cms.Process("L1DTTrigPhase2Prod")

process.load('Configuration.Geometry.GeometryExtended2026D49Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2026D49_cff')

process.load("L1Trigger.DTTriggerPhase2.dtTriggerPhase2PrimitiveDigis_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff")

process.DTGeometryESModule.applyAlignment = True

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_data', '')

process.load("L1Trigger.DTTriggerPhase2.CalibratedDigis_cfi")
process.load("L1Trigger.DTTriggerPhase2.dtTriggerPhase2PrimitiveDigis_cfi")
process.source = cms.Source("PoolSource",fileNames = cms.untracked.vstring('file:/tmp/carrillo/digis_segments_Run2016BSingleMuonRAW-RECO.root'))

#test tunning
#process.dtTriggerPhase2PrimitiveDigis.tanPhiTh = cms.untracked.double(1.4) #temporary for eta
process.dtTriggerPhase2PrimitiveDigis.global_coords_filename = cms.FileInPath('L1Trigger/DTTriggerPhase2/data/run2_global_coord_perp_x_phi0.txt')#for run2 data
process.dtTriggerPhase2PrimitiveDigis.cmssw_for_global = cms.untracked.bool(True) #forcing cmssw coordinates transformation

#data scenario 1
process.dtTriggerPhase2PrimitiveDigis.scenario = 1
process.CalibratedDigis.scenario = 1



process.out = cms.OutputModule("PoolOutputModule",
                               outputCommands = cms.untracked.vstring('keep *'),
                               fileName = cms.untracked.string('/tmp/carrillo/DTTriggerPhase2Primitives.root')
)

process.p = cms.Path(process.CalibratedDigis*process.dtTriggerPhase2PrimitiveDigis)
process.this_is_the_end = cms.EndPath(process.out)


####################################################################




#process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1000))
#process.dtTriggerPhase2PrimitiveDigis.dump = False
#process.dtTriggerPhase2PrimitiveDigis.debug = False
#process.dtTriggerPhase2PrimitiveDigis.chi2Th = cms.untracked.double(0.16)

#process.load("FWCore.MessageLogger.MessageLogger_cfi")
#process.MessageLogger = cms.Service("MessageLogger",
#        destinations = cms.untracked.vstring("detailedInfo"),
#        detailedInfo = cms.untracked.PSet(threshold = cms.untracked.string("INFO"),
#        categories = cms.untracked.vstring("DTTrigPhase2Prod"),
#        extension = cms.untracked.string(".txt")),
#        debugModules = cms.untracked.vstring("dtTriggerPhase2PrimitiveDigis"),
#)
####################### SliceTest specials ##############################

#Chi2 -> Changing a lot lately
#process.dtTriggerPhase2PrimitiveDigis.chi2Th = cms.untracked.double(0.16)

#LSB -> Position 0.025 cm instead of 0.004 cm
#process.dtTriggerPhase2PrimitiveDigis.use_LSB = True
#process.dtTriggerPhase2PrimitiveDigis.x_precision = cms.untracked.double(1./(10.*16.))
#process.dtTriggerPhase2PrimitiveDigis.x_precision = cms.untracked.double(0.025)
#process.dtTriggerPhase2PrimitiveDigis.tanPsi_precision = cms.untracked.double(1./4096.)

#Correlate with BX
#process.dtTriggerPhase2PrimitiveDigis.useBX_correlation = True
#process.dtTriggerPhase2PrimitiveDigis.dBX_correlate_TP = 1

#Correlate with tanPsi
#process.dtTriggerPhase2PrimitiveDigis.dTanPsi_correlate_TP = cms.untracked.double(9999./4096.)
#process.dtTriggerPhase2PrimitiveDigis.dTanPsi_correlate_TP = cms.untracked.double(900./4096.)

#Confirmation forbidden
#process.dtTriggerPhase2PrimitiveDigis.allow_confirmation = True

#TanPsi stuff
