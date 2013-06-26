import FWCore.ParameterSet.Config as cms

process = cms.Process("BTagDemooo")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")

process.load("RecoBTag.Configuration.RecoBTag_cff")

process.source = cms.Source("PoolSource",
	fileNames = cms.untracked.vstring('file:/home/bilibao/CMSSW/CMSSW_2007-05-14/src/RecoBTag/ImpactParameter/test/btag.root')
)

process.maxEvents = cms.untracked.PSet(
	input = cms.untracked.int32(-1)
)

process.out = cms.OutputModule("PoolOutputModule",
	fileName = cms.untracked.string('/tmp/aaabb.root')
)

process.p = cms.Path(process.btagging)

process.outpath = cms.EndPath(process.out)
