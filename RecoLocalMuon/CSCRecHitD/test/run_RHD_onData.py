import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("EventFilter.CSCRawToDigi.cscFrontierCablingUnpck_cff")

process.load("EventFilter.CSCRawToDigi.cscUnpacker_cfi")

process.load("Geometry.MuonCommonData.muonIdealGeometryXML_cfi")

process.load("Geometry.CSCGeometry.cscGeometry_cfi")

process.load("CalibMuon.Configuration.CSC_FrontierDBConditions_DevDB_cff")

process.load("RecoLocalMuon.CSCRecHitD.cscRecHitD_cfi")

process.load("RecoLocalMuon.CSCSegment.cscSegments_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(101)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('dcache:/pnfs/cms/WAX/11/store/data/MTCC/pass1/3792/A/mtcc.00003792.A.testStorageManager_0.0.root')
)

process.MuonNumberingInitialization = cms.ESProducer("MuonNumberingInitialization")

process.Timing = cms.Service("Timing")

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('data.root')
)

process.p = cms.Path(process.muonCSCDigis*process.csc2DRecHits)
process.muonCSCDigis.isMTCCData = True
process.muonCSCDigis.UseExaminer = False
process.cscSegments.algo_type = 4


