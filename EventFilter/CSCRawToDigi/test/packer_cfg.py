import FWCore.ParameterSet.Config as cms
# packs and unpacks data from a dataset which already has digis
process = cms.Process("ANAL")
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'MC_31X_V9::All'


process.load("EventFilter.CSCRawToDigi.cscPacker_cfi")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/relval/CMSSW_3_4_0_pre1/RelValSingleMuPt100/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0008/D670F304-B7B5-DE11-A234-001D09F2983F.root')
)
process.maxEvents = cms.untracked.PSet(
     input = cms.untracked.int32(50)
     )


process.load("EventFilter.CSCRawToDigi.cscUnpacker_cfi")
process.DQMStore = cms.Service("DQMStore")
process.load("SimMuon.CSCDigitizer.cscDigiDump_cfi")
process.muonCSCDigis.InputObjects = "rawDataCollector"
process.muonCSCDigis.Debug = True
process.muonCSCDigis.UseExaminer = False
process.p1 = cms.Path(process.cscpacker+process.muonCSCDigis+process.cscDigiDump)


