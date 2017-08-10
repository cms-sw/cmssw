import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

#define HltBTagPostValidation for the b-tag DQM validation (efficiency and mistagrate plot)
HltBTagPostValidation = DQMEDHarvester("HLTBTagHarvestingAnalyzer",
	HLTPathNames = cms.vstring(
	'HLT_PFMET120_',
	'HLT_HT300PT30_QuadJet_75_60_45_40_v',
	'HLT_PFHT380_SixJet32_DoubleBTagCSV_',
	'HLT_Ele28_eta2p1_WPTight_Gsf_HT150_',
	'HLT_IsoMu24_eta2p1_'
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

