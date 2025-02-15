import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

#define HltBTagPostValidation for the b-tag DQM validation (efficiency and mistagrate plot)
HltBTagPostValidation = DQMEDHarvester("HLTBTagHarvestingAnalyzer",
        mainFolder   = cms.string("HLT/BTV/Validation"),
	HLTPathNames = cms.vstring(
	    'HLT_PFMET120_PFMHT120_IDTight_v',
	    'HLT_PFHT330PT30_QuadPFJet_75_60_45_40_v',
	    'HLT_PFHT400_SixPFJet32_PNet2BTagMean0p50_v',
	    'HLT_PFHT450_SixPFJet36_PNetBTag0p35_v',  
	    'HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_PFDiJet30_v',
	    'HLT_BTagMu_AK4DiJet20_Mu5_v',
	    'HLT_BTagMu_AK4DiJet20_Mu5_v',
	    'HLT_BTagMu_AK4DiJet20_Mu5_v',
	),
        histoName = cms.vstring(
            'hltParticleNetDiscriminatorsJetTags',
            'hltParticleNetDiscriminatorsJetTags',
            'hltParticleNetDiscriminatorsJetTags',
            'hltParticleNetDiscriminatorsJetTags',
            'hltParticleNetDiscriminatorsJetTags',
            'hltBSoftMuonDiJet20L1FastJetL25Jets',
            'hltDeepJetDiscriminatorsJetTags',
            'hltParticleNetDiscriminatorsJetTags',
	),
	minTag	= cms.double(0.2), #Medium WP for 2023, see https://btv-wiki.docs.cern.ch/ScaleFactors/Run3Summer23/
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

