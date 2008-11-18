import FWCore.ParameterSet.Config as cms

process = cms.Process("RecHitReader")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("Configuration.StandardSequences.Geometry_cff")

process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("Configuration.StandardSequences.FakeConditions_cff")

process.load("Configuration.StandardSequences.Reconstruction_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/relval/2008/6/22/RelVal-RelValSingleMuPt10-1214048167-IDEAL_V2-2nd/0004/0AE2B3E3-0141-DD11-846F-000423D98BC4.root')
)

process.Timing = cms.Service("Timing")

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

process.ReadLocalMeasurement = cms.EDFilter("ReadRecHit",
    VerbosityLevel = cms.untracked.int32(1),
    RecHitProducer = cms.string('siStripMatchedRecHits')
)

process.p = cms.Path(process.striptrackerlocalreco*process.ReadLocalMeasurement)


