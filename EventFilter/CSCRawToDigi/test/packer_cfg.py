import FWCore.ParameterSet.Config as cms
# packs and unpacks data from a dataset which already has digis
process = cms.Process("ANAL")
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'IDEAL_30X::All'


process.load("EventFilter.CSCRawToDigi.cscPacker_cfi")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/relval/CMSSW_3_1_0_pre5/RelValSingleMuPt100/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0000/EC0724B2-AC2B-DE11-BDB4-000423D991F0.root')
)
process.maxEvents = cms.untracked.PSet(
     input = cms.untracked.int32(1000)
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


