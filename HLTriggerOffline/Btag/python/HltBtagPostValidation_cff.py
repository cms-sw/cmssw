import FWCore.ParameterSet.Config as cms

#define HltBTagPostValidation for the b-tag DQM validation (efficiency and mistagrate plot)
HltBTagPostValidation = cms.EDAnalyzer("HLTBTagHarvestingAnalyzer",
	HLTPathNames = cms.vstring('HLT_PFMET120_NoiseCleaned_BTagCSV07_'),
	histoName	= cms.vstring('hltCombinedSecondaryVertexBJetTagsCalo'),
	minTag	= cms.double(0.6),
	# MC stuff
	mcFlavours = cms.PSet(
		light = cms.vuint32(1, 2, 3, 21), # udsg
		c = cms.vuint32(4),
		b = cms.vuint32(5),
		g = cms.vuint32(21),
		uds = cms.vuint32(1, 2, 3)
	)
)

#put all in a path
HltBTagPostVal = cms.Sequence(
	HltBTagPostValidation
)

