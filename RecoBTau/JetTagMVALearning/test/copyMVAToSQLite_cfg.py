import FWCore.ParameterSet.Config as cms

process = cms.Process("MVAJetTagsSQLiteSave")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

process.calib = cms.ESSource("BTauGenericMVAJetTagComputerFileSource",
#	ImpactParameterMVA = cms.string('ImpactParameterMVA.mva'), 
#	CombinedMVA = cms.string('CombinedMVA.mva'),
	CombinedSVRecoVertex = cms.string('CombinedSVRecoVertex.mva'), 
	CombinedSVPseudoVertex = cms.string('CombinedSVPseudoVertex.mva'), 
	CombinedSVNoVertex = cms.string('CombinedSVNoVertex.mva'), 
#	CombinedSVMVARecoVertex = cms.string('CombinedSVMVARecoVertex.mva'), 
#	CombinedSVMVAPseudoVertex = cms.string('CombinedSVMVAPseudoVertex.mva'), 
#	CombinedSVMVANoVertex = cms.string('CombinedSVMVANoVertex.mva'),
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
	BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
	DBParameters = cms.PSet( messageLevel = cms.untracked.int32(0) ),
	timetype = cms.untracked.string('runnumber'),
	connect = cms.string('sqlite_file:MVAJetTags.db'),
	toPut = cms.VPSet(cms.PSet(
		record = cms.string('BTauGenericMVAJetTagComputerRcd'),
		tag = cms.string('MVAJetTags_CMSSW_4_4_4')
	))
)

process.jetTagMVATrainerSave = cms.EDAnalyzer("JetTagMVATrainerSave",
	toPut = cms.vstring(),
	toCopy = cms.vstring(
#		'ImpactParameterMVA', 
#		'CombinedMVA', 
		'CombinedSVRecoVertex', 
		'CombinedSVPseudoVertex', 
		'CombinedSVNoVertex', 
#		'CombinedSVMVARecoVertex', 
#		'CombinedSVMVAPseudoVertex', 
#		'CombinedSVMVANoVertex'
	)
)

process.outpath = cms.EndPath(process.jetTagMVATrainerSave)
