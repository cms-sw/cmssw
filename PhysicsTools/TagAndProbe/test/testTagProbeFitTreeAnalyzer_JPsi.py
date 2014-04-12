import FWCore.ParameterSet.Config as cms

process = cms.Process("TagProbe")

process.load('FWCore.MessageService.MessageLogger_cfi')

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )    

process.load("PhysicsTools.TagAndProbe.TagProbeFitTreeAnalyzer_cfi")

process.fitness = cms.Path(
    process.TagProbeFitTreeAnalyzer
)

