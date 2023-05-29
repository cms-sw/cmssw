import FWCore.ParameterSet.Config as cms

process = cms.Process("RecHitReader")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FakeConditions_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/relval/CMSSW_2_1_0/RelValSingleMuPt10/GEN-SIM-DIGI-RECO/SpecialConditions_v1/0001/0A2D1D45-0D61-DD11-937B-001BFCDBD154.root')
)

process.Timing = cms.Service("Timing")

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

process.ReadLocalMeasurement = cms.EDAnalyzer("ReadRecHit",
    VerbosityLevel = cms.untracked.int32(1),
    RecHitProducer = cms.string('siStripMatchedRecHits')
)

process.p = cms.Path(process.striptrackerlocalreco*process.ReadLocalMeasurement)


