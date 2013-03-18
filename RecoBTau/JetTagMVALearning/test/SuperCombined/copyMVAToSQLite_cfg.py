import FWCore.ParameterSet.Config as cms

process = cms.Process("MVAJetTagsSQLiteSave")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

process.calib = cms.ESSource("BTauGenericMVAJetTagComputerFileSource",
##	ImpactParameterMVA = cms.string('ImpactParameterMVA.mva'), 
	CombinedMVA = cms.string('SC_woJP/SC_weights.mva'),
#	CombinedSVRecoVertex = cms.string('/user/gvonsem/BTagServiceWork/MVA/testBtagVal_NewJuly2012/CMSSW_5_3_4_patch1/src/RecoBTau/JetTagMVALearning/test/CSV_LR_withPFnoPU_Adding2HighPtBins_nonFit-ReweightingTESTS/OffsetFix/CombinedSVRecoVertex.mva'), 
#	CombinedSVPseudoVertex = cms.string('/user/gvonsem/BTagServiceWork/MVA/testBtagVal_NewJuly2012/CMSSW_5_3_4_patch1/src/RecoBTau/JetTagMVALearning/test/CSV_LR_withPFnoPU_Adding2HighPtBins_nonFit-ReweightingTESTS/OffsetFix/CombinedSVPseudoVertex.mva'), 
#	CombinedSVNoVertex = cms.string('/user/gvonsem/BTagServiceWork/MVA/testBtagVal_NewJuly2012/CMSSW_5_3_4_patch1/src/RecoBTau/JetTagMVALearning/test/CSV_LR_withPFnoPU_Adding2HighPtBins_nonFit-ReweightingTESTS/OffsetFix/CombinedSVNoVertex.mva'), 
#	CombinedSVRecoVertex = cms.string('/user/gvonsem/BTagServiceWork/MVA/testBtagVal_NewJuly2012/CMSSW_5_3_4_patch1/src/RecoBTau/JetTagMVALearning/test/CSV_LR_withPFnoPU_Adding2HighPtBins_nonFit-ReweightingTESTS/CombinedSVRecoVertex.mva'), 
#	CombinedSVPseudoVertex = cms.string('/user/gvonsem/BTagServiceWork/MVA/testBtagVal_NewJuly2012/CMSSW_5_3_4_patch1/src/RecoBTau/JetTagMVALearning/test/CSV_LR_withPFnoPU_Adding2HighPtBins_nonFit-ReweightingTESTS/CombinedSVPseudoVertex.mva'), 
#	CombinedSVNoVertex = cms.string('/user/gvonsem/BTagServiceWork/MVA/testBtagVal_NewJuly2012/CMSSW_5_3_4_patch1/src/RecoBTau/JetTagMVALearning/test/CSV_LR_withPFnoPU_Adding2HighPtBins_nonFit-ReweightingTESTS/CombinedSVNoVertex.mva'), 
	CombinedSVRecoVertex = cms.string('/user/pvmulder/NewEraOfDataAnalysis/BTagServiceWork/CMSSW_5_3_4_patch1/src/CombinedSVRecoVertex_2HighPtBins.mva'), 
	CombinedSVPseudoVertex = cms.string('/user/pvmulder/NewEraOfDataAnalysis/BTagServiceWork/CMSSW_5_3_4_patch1/src/CombinedSVPseudoVertex_2HighPtBins.mva'), 
	CombinedSVNoVertex = cms.string('/user/pvmulder/NewEraOfDataAnalysis/BTagServiceWork/CMSSW_5_3_4_patch1/src/CombinedSVNoVertex_2HighPtBins.mva'), 
	CombinedSVMVARecoVertex = cms.string('/user/pvmulder/NewEraOfDataAnalysis/BTagServiceWork/CMSSW_5_3_4_patch1/src/CombinedSVMVARecoVertex.mva'), 
	CombinedSVMVAPseudoVertex = cms.string('/user/pvmulder/NewEraOfDataAnalysis/BTagServiceWork/CMSSW_5_3_4_patch1/src/CombinedSVMVAPseudoVertex.mva'), 
	CombinedSVMVANoVertex = cms.string('/user/pvmulder/NewEraOfDataAnalysis/BTagServiceWork/CMSSW_5_3_4_patch1/src/CombinedSVMVANoVertex.mva'),
	GhostTrackRecoVertex = cms.string('/user/pvmulder/NewEraOfDataAnalysis/BTagServiceWork/CMSSW_5_3_4_patch1/src/GhostTrackRecoVertex.mva'), 
	GhostTrackPseudoVertex = cms.string('/user/pvmulder/NewEraOfDataAnalysis/BTagServiceWork/CMSSW_5_3_4_patch1/src/GhostTrackPseudoVertex.mva'), 
	GhostTrackNoVertex = cms.string('/user/pvmulder/NewEraOfDataAnalysis/BTagServiceWork/CMSSW_5_3_4_patch1/src/GhostTrackNoVertex.mva'),
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
	BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
	DBParameters = cms.PSet( messageLevel = cms.untracked.int32(0) ),
	timetype = cms.untracked.string('runnumber'),
	connect = cms.string('sqlite_file:SC_weights.db'),
	toPut = cms.VPSet(cms.PSet(
		record = cms.string('BTauGenericMVAJetTagComputerRcd'),
		tag = cms.string('MVAJetTags_CMSSW_5_3_4')
	))
)

process.jetTagMVATrainerSave = cms.EDAnalyzer("JetTagMVATrainerSave",
	toPut = cms.vstring(),
	toCopy = cms.vstring(
##		'ImpactParameterMVA', 
		'CombinedMVA', 
#		'CombinedSVRetrain2RecoVertex', 
#		'CombinedSVRetrain2PseudoVertex', 
#		'CombinedSVRetrain2NoVertex', 
		'CombinedSVRecoVertex', 
		'CombinedSVPseudoVertex', 
		'CombinedSVNoVertex', 
		'CombinedSVMVARecoVertex', 
		'CombinedSVMVAPseudoVertex', 
		'CombinedSVMVANoVertex',
#		'GhostTrackRecoVertex', 
#		'GhostTrackPseudoVertex', 
#		'GhostTrackNoVertex'
#		'CombinedSVRetrain2RecoVertex', 
#		'CombinedSVRetrain2PseudoVertex', 
#		'CombinedSVRetrain2NoVertex', 
	)
)

process.outpath = cms.EndPath(process.jetTagMVATrainerSave)
