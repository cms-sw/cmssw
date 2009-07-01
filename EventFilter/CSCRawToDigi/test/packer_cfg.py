import FWCore.ParameterSet.Config as cms
# packs and unpacks data from a dataset which already has digis
process = cms.Process("ANAL")
process.load("EventFilter.CSCRawToDigi.cscFrontierCablingPack_cff")
process.load("CalibMuon.Configuration.CSC_FakeDBConditions_cff")
process.load("EventFilter.CSCRawToDigi.cscFrontierCablingUnpck_cff")


process.load("EventFilter.CSCRawToDigi.cscPacker_cfi")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:simevent.root')
)

process.load("EventFilter.CSCRawToDigi.cscUnpacker_cfi")
process.DQMStore = cms.Service("DQMStore")

process.dump = cms.EDFilter("CSCDigiDump",
    wireDigiTag = cms.InputTag("muonCSCDigis","MuonCSCWireDigi"),
    empt = cms.InputTag(""),
    stripDigiTag = cms.InputTag("muonCSCDigis","MuonCSCStripDigi"),
    comparatorDigiTag = cms.InputTag("muonCSCDigis","MuonCSCComparatorDigi")
)

process.muonCSCDigis.InputObjects = "rawDataCollector"
process.muonCSCDigis.Debug = True
process.muonCSCDigis.UseExaminer = False
process.p1 = cms.Path(process.cscpacker+process.muonCSCDigis+process.dump)


