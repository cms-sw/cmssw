#define bTagValidation for the b-tag DQM validation (distribution plot)
process.bTagValidation = cms.EDAnalyzer("HLTBTagPerformanceAnalyzer",
	TriggerResults = cms.InputTag('TriggerResults'),
	HLTPathNames = cms.vstring('HLT_PFMHT100_SingleCentralJet60_BTagCSV0p6_v1'),
	JetTag = cms.VInputTag(cms.InputTag("hltL3CombinedSecondaryVertexBJetTags")),
	MinJetPT = cms.double(20),
	mcFlavours = cms.PSet(
		light = cms.vuint32(1, 2, 3, 21), # udsg
		c = cms.vuint32(4),
		b = cms.vuint32(5),
		g = cms.vuint32(21),
		uds = cms.vuint32(1, 2, 3)
	),
	mcPartons = cms.InputTag("hltJetsbyValAlgo")
)



