#!/usr/bin/env cmsRun
import FWCore.ParameterSet.Config as cms

process = cms.Process("MatchingAnalysis")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'INFO'

process.source = cms.Source("PoolSource",
	fileNames = cms.untracked.vstring('file:test.root'),
	skipEvents = cms.untracked.uint32(0)
)

process.TFileService = cms.Service("TFileService",
	fileName = cms.string('lheAnalysis.root')
)

process.lheAnalyzer = cms.EDAnalyzer("LHEAnalyzer",
	src = cms.InputTag('VtxSmeared'),

	jetInput = cms.PSet(
		partonicFinalState = cms.bool(True),
		excludedResonances = cms.vuint32(6),
		excludedFromResonances = cms.vuint32(1, 2, 3, 4, 5, 21, 24),
		onlyHardProcess = cms.bool(True),
		tausAsJets = cms.bool(False),
	),

	jetClustering = cms.PSet(
		name = cms.string('SISCone'),
		coneOverlapThreshold = cms.double(0.75),
		splitMergeScale = cms.string('pttilde'),
		maxPasses = cms.int32(0),
		protojetPtMin = cms.double(0.0),
		caching = cms.bool(False)
	),

	defaultDeltaCut	= cms.double(0.7),
	defaultPtCut	= cms.double(70.0),
	maxEta		= cms.double(5.0),
	useEt		= cms.bool(True),
	binsDelta	= cms.uint32(50),
	minDelta	= cms.double(0.2),
	maxDelta	= cms.double(1.2),
	binsPt		= cms.uint32(100),
	minPt		= cms.double(10.0),
	maxPt		= cms.double(1000.0),
	minDJR		= cms.uint32(0),
	maxDJR		= cms.uint32(5),
)

process.p = cms.Path(process.lheAnalyzer)
