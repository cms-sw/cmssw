import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

#define HltBTagPostValidation for the b-tag DQM validation (efficiency and mistagrate plot)
HltBTagPostValidation = DQMEDHarvester("HLTBTagHarvestingAnalyzer",
	HLTPathNames = cms.vstring(
	'HLT_PFMET120_',
	'HLT_CaloMHTNoPU90_PFMET90_PFMHT90_IDTight_',
	'HLT_QuadPFJet_VBF',
	'HLT_Ele32_eta2p1_',
	'HLT_IsoMu21_eta2p1_'
	),
	histoName	= cms.vstring(
	'hltCombinedSecondaryVertexBJetTagsCalo',
	'hltCombinedSecondaryVertexBJetTagsCalo',
	'hltCombinedSecondaryVertexBJetTagsCalo',
	'hltCombinedSecondaryVertexBJetTagsPF',
	'hltCombinedSecondaryVertexBJetTagsPF',
	),
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

