import FWCore.ParameterSet.Config as cms

process = cms.Process("IPTrainer")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.combinedMVATrainer = cms.EDAnalyzer("JetTagMVATreeTrainer",
	useCategories		= cms.bool(False),
	calibrationRecord	= cms.string("combinedMVA"),
	ignoreFlavours		= cms.vint32(0),
	signalFlavours		= cms.vint32(5, 7),
	minimumTransverseMomentum = cms.double(15.0),
	minimumPseudoRapidity	= cms.double(0),
	maximumPseudoRapidity	= cms.double(2.5),

	factor = cms.double(50),
	bound = cms.double(50),

	fileNames = cms.vstring(
		"/afs/cern.ch/work/p/pvmulder/public/BTagging/FINAL_NOSPIKES/CMSSW_5_3_4_patch1/src/combinedMVA_B.root",
	),
	weightFile = cms.string("./weights/combinedMVA_B_histo.txt"),
	biasFiles = cms.vstring(
		"-",
		"-",
		"*"
	),
	#maxEvents=cms.untracked.int32(10)
)

process.looper = cms.Looper("JetTagMVATrainerLooper",
	trainers = cms.VPSet(
		cms.PSet(
			calibrationRecord = cms.string("combinedMVA"),
			trainDescription = cms.untracked.string("Save_B.xml"),
			loadState = cms.untracked.bool(False),
			saveState = cms.untracked.bool(False)
		)
	)
)

process.p = cms.Path(process.combinedMVATrainer)
