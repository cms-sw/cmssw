import FWCore.ParameterSet.Config as cms

process = cms.Process("MVAJetTagsSQLiteSave")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

process.load("RecoBTau.Configuration.RecoBTau_FrontierConditions_IntDB_cff")
process.BTauMVAJetTagComputerRecord.toGet = cms.VPSet(cms.PSet(
	record = cms.string('BTauGenericMVAJetTagComputerRcd'),
	tag = cms.string('MVAJetTags_CMSSW_2_0_0_mc')
))

process.prefer("BTauMVAJetTagComputerRecord")

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
	BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
	DBParameters = cms.PSet( messageLevel = cms.untracked.int32(0) ),
	timetype = cms.untracked.string('runnumber'),
	connect = cms.string('sqlite_file:MVAJetTagsFakeConditions.db'),
	toPut = cms.VPSet(cms.PSet(
		record = cms.string('BTauGenericMVAJetTagComputerRcd'),
		tag = cms.string('MVAJetTags_CMSSW_2_0_0_mc')
	))
)

process.jetTagMVATrainerSave = cms.EDFilter("JetTagMVATrainerSave",
	toPut = cms.vstring(),
	toCopy = cms.vstring(
		'ImpactParameterMVA', 
		'CombinedSVRecoVertex', 
		'CombinedSVPseudoVertex', 
		'CombinedSVNoVertex', 
		'CombinedSVMVARecoVertex', 
		'CombinedSVMVAPseudoVertex', 
		'CombinedSVMVANoVertex'
	)
)

process.outpath = cms.EndPath(process.jetTagMVATrainerSave)
