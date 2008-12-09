
import FWCore.ParameterSet.Config as cms

process = cms.Process("CastorProducts")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)
process.source = cms.Source("PoolSource",
    fileNames = 
cms.untracked.vstring('file:digiout.root')
)

process.fromRecHits = cms.EDFilter("Castor",
    FullReco = cms.untracked.bool(True),
    Egamma_maxDepth = cms.untracked.double(14488.0),
    Egamma_maxWidth = cms.untracked.double(0.2),
    KtRecombination = cms.untracked.uint32(2),
    Egamma_minRatio = cms.untracked.double(0.5),
    KtrParameter = cms.untracked.double(1.0),
    towercut = cms.untracked.double(0.)
)

process.MyOutputModule = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('keep *_fromRecHits_*_*'),
    fileName = cms.untracked.string('recooutput.root')
)

process.producer = cms.Path(process.fromRecHits)
process.end = cms.EndPath(process.MyOutputModule)

