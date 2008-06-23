import FWCore.ParameterSet.Config as cms

process = cms.Process("VerySimplePATAnalysis")
process.load("PhysicsTools.StarterKit.PatAnalyzerSkeleton_cfi")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:myPATfile.root')
)

process.MessageLogger = cms.Service("MessageLogger")

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('histo.root')
)

process.p = cms.Path(process.verySimplePATAnalysis)

