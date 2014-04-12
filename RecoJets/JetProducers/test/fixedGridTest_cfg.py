import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/relval/CMSSW_4_2_1/RelValTTbar/GEN-SIM-RECO/START42_V10-v1/0025/72E96F34-C166-E011-9556-003048678FA6.root')
)

process.load("RecoJets.JetProducers.fixedGridRhoProducer_cfi")

process.RECO = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('test.root')
)
process.p1 = cms.Path(process.fixedGridRhoCentral*process.fixedGridRhoForward*process.fixedGridRhoAll)
process.outpath = cms.EndPath(process.RECO)
