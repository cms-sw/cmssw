#!/usr/bin/env cmsRun

import FWCore.ParameterSet.Config as cms

process = cms.Process("test1")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")

process.load("RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi")
process.load("RecoJets.JetAssociationProducers.ak5JTA_cff")

process.load("RecoBTau.JetTagComputer.jetTagRecord_cfi")
process.load("RecoBTag.ImpactParameter.impactParameter_cff")

process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.source = cms.Source("PoolSource",
	fileNames = cms.untracked.vstring('file:ttbar.root')
)

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
	process.CondDBSetup,
	timetype = cms.string('runnumber'),
	toGet = cms.VPSet(
		cms.PSet(
			record = cms.string('BTauGenericMVAJetTagComputerRcd'),
			tag = cms.string('test_tag')
		)
	),
	connect = cms.string('sqlite_file:MVATestJetTags.db'),
	BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService')
)

process.Output = cms.OutputModule("PoolOutputModule",
	fileName = cms.untracked.string('file:test.root'),
	outputCommands = cms.untracked.vstring(
		'drop *', 
		'keep recoJetTags_*_*_*', 
		'keep *_ak5JetTracksAssociatorAtVertex_*_*', 
		'keep *_ak5CaloJets_*_*', 
		'keep *_trackCountingJetTags_*_*', 
		'keep *_impactParameterTagInfos_*_*', 
		'keep recoTrackExtras_*_*_*', 
		'keep recoTracks_*_*_*', 
		'keep recoVertexs_*_*_*')
)

process.p = cms.Path(
	process.offlinePrimaryVertices *
	process.ak5JetTracksAssociatorAtVertex *
	process.impactParameterTagInfos *
	process.impactParameterMVABJetTags
)

process.outpath = cms.EndPath(process.Output)
