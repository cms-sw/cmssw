import FWCore.ParameterSet.Config as cms

process = cms.Process("IPTrainer")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load("RecoBTau.JetTagComputer.jetTagRecord_cfi")
process.load("RecoBTag.ImpactParameter.impactParameterMVAComputer_cfi")
process.load("PhysicsTools.JetMCAlgos.CaloJetsMCFlavour_cfi")

process.source = cms.Source("PoolSource",
	fileNames = cms.untracked.vstring('file:testTagInfos.root')
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
	BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
	DBParameters = cms.PSet( messageLevel = cms.untracked.int32(0) ),
	timetype = cms.untracked.string('runnumber'),
	connect = cms.string('sqlite_file:MVAImpactParameterJetTags.db'),
	toPut = cms.VPSet(cms.PSet(
		record = cms.string('BTauGenericMVAJetTagComputerRcd'),
		tag = cms.string('MVAJetTags_CMSSW_2_0_0_mc')
	))
)

process.jetTagMVATrainerSave = cms.EDAnalyzer("JetTagMVATrainerSave",
	toPut = cms.vstring('ImpactParameter'),
	toCopy = cms.vstring()
)

process.impactParameterMVATrainer = cms.EDAnalyzer("JetTagMVATrainer",
	minimumTransverseMomentum = cms.double(10.0),
	tagInfos = cms.VInputTag(cms.InputTag("impactParameterTagInfos")),
	useCategories = cms.bool(False),
	calibrationRecord = cms.string('ImpactParameter'),
	maximumPseudoRapidity = cms.double(2.5),
	signalFlavours = cms.vint32(5, 7),
	minimumPseudoRapidity = cms.double(0.0),
	jetTagComputer = cms.string('impactParameterMVAComputer'),
	jetFlavourMatching = cms.InputTag("AK5byValAlgo"),
	ignoreFlavours = cms.vint32(0)
)

process.looper = cms.Looper("JetTagMVATrainerLooper",
	trainers = cms.VPSet(cms.PSet(
		calibrationRecord = cms.string('ImpactParameter'),
		saveState = cms.untracked.bool(False),
		trainDescription = cms.untracked.string('ImpactParameterMVATrainer.xml'),
		loadState = cms.untracked.bool(False)
	))
)

process.mcAlgoJetFlavour = cms.Sequence(
	process.myPartons *
	process.AK5byRef *
	process.AK5byValAlgo
)

process.p = cms.Path(
	process.mcAlgoJetFlavour *
	process.impactParameterMVATrainer
)

process.outpath = cms.EndPath(process.jetTagMVATrainerSave)
