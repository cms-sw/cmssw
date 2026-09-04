import FWCore.ParameterSet.Config as cms
from HLTriggerOffline.Btag.hltBtagJetMCTools_cff import *

#denominator trigger
hltBtagTriggerSelection = cms.EDFilter( "TriggerResultsFilter",
    triggerConditions = cms.vstring(
        "HLT_PFMET120_PFMHT120_IDTight_v* OR HLT_PFHT330PT30_QuadPFJet_75_60_45_40_v* OR HLT_PFHT400_SixPFJet32_PNet2BTag* OR HLT_IsoMu24_eta2p1_v*"),
    hltResults = cms.InputTag( "TriggerResults", "", "HLT" ),
#    l1tResults = cms.InputTag( "simGtDigis" ),
    l1tResults = cms.InputTag( "" ),
    throw = cms.bool( False )
)

#correct the jet used for the matching
hltBtagJetsbyRef.jets = cms.InputTag("ak4GenJetsNoNu")

#define HltVertexValidationVertices for the vertex DQM validation
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
HltVertexValidationVertices= DQMEDAnalyzer('HLTVertexPerformanceAnalyzer',
        SimVertexCollection = cms.InputTag("g4SimHits"),
	TriggerResults = cms.InputTag('TriggerResults','',"HLT"),
	mainFolder   = cms.string("HLT/BTV/Validation"),
	HLTPathNames =cms.vstring(
	    'HLT_PFMET120_PFMHT120_IDTight_v',
	    'HLT_PFHT330PT30_QuadPFJet_75_60_45_40_v',
	    'HLT_PFHT400_SixPFJet32_PNet2BTagMean0p50_v',
	    'HLT_PFHT450_SixPFJet36_PNetBTag0p35_v',  
	    'HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_PFDiJet30_v',
	    'HLT_BTagMu_AK4DiJet20_Mu5_v',
	    'HLT_BTagMu_AK4DiJet20_Mu5_v',
	    'HLT_BTagMu_AK4DiJet20_Mu5_v',
	),
	Vertex = cms.VInputTag(
		cms.InputTag("hltVerticesPF"), 
	)
)

#define bTagValidation for the b-tag DQM validation (distribution plot)
hltbTagValidation = DQMEDAnalyzer('HLTBTagPerformanceAnalyzer',
	TriggerResults = cms.InputTag('TriggerResults','','HLT'),
	mainFolder   = cms.string("HLT/BTV/Validation"),
	HLTPathNames =cms.vstring(
	    'HLT_PFMET120_PFMHT120_IDTight_v',
	    'HLT_PFHT330PT30_QuadPFJet_75_60_45_40_v',
	    'HLT_PFHT400_SixPFJet32_PNet2BTagMean0p50_v',
	    'HLT_PFHT450_SixPFJet36_PNetBTag0p35_v',  
	    'HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_PFDiJet30_v',
	    'HLT_BTagMu_AK4DiJet20_Mu5_v',
	    'HLT_BTagMu_AK4DiJet20_Mu5_v',
	    'HLT_BTagMu_AK4DiJet20_Mu5_v',

	),
	JetTag = cms.VInputTag(
	    cms.InputTag("hltParticleNetDiscriminatorsJetTags", "BvsAll"),
	    cms.InputTag("hltParticleNetDiscriminatorsJetTags", "BvsAll"),
	    cms.InputTag("hltParticleNetDiscriminatorsJetTags", "BvsAll"),
	    cms.InputTag("hltParticleNetDiscriminatorsJetTags", "BvsAll"),
	    cms.InputTag("hltParticleNetDiscriminatorsJetTags", "BvsAll"),
	    cms.InputTag("hltBSoftMuonDiJet20L1FastJetL25Jets"),
	    cms.InputTag("hltDeepJetDiscriminatorsJetTags", "BvsAll"),
	    cms.InputTag("hltParticleNetDiscriminatorsJetTags", "BvsAll"),
	),
	MinJetPT = cms.double(20),
	mcFlavours = cms.PSet(
		light = cms.vuint32(1, 2, 3, 21), # udsg
		c = cms.vuint32(4),
		b = cms.vuint32(5),
		g = cms.vuint32(21),
		uds = cms.vuint32(1, 2, 3)
	),
	mcPartons = cms.InputTag("hltBtagJetsbyValAlgo")
)

#put all in a path
hltbtagValidationSequence = cms.Sequence(
#	remove noisy warnings
#	hltBtagTriggerSelection +
	hltBtagJetMCTools +
	HltVertexValidationVertices +
	hltbTagValidation
)

# fastsim customs
from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toModify(HltVertexValidationVertices, SimVertexCollection = "fastSimProducer")
    # are these customs actually needed?
    #HltVertexValidationVertices.HLTPathNames =cms.vstring(
    #'HLT_PFMET120_NoiseCleaned_BTagCSV07_v',
    #'HLT_PFMET120_NoiseCleaned_BTagCSV07_v',
    #	'HLT_CaloMHTNoPU90_PFMET90_PFMHT90_IDLoose_',
    #	'HLT_CaloMHTNoPU90_PFMET90_PFMHT90_IDLoose_',
    #	'HLT_QuadPFJet_VBF',
    #	'HLT_QuadPFJet_VBF',
    #	'HLT_Ele32_eta2p1_',
    #	'HLT_IsoMu24_eta2p1_')
    #HltVertexValidationVertices.Vertex = cms.VInputTag(
    #    cms.InputTag("hltVerticesL3"), 
    #    cms.InputTag("hltFastPVPixelVertices"),
    #    cms.InputTag("hltVerticesPF"), 
    #)

#Including phase2 conditions
from Configuration.Eras.Modifier_phase2_common_cff import phase2_common

triggerConditions_phase2 = cms.vstring(
    "HLT_DoublePFPuppiJets128_DoublePFPuppiBTagDeepCSV_2p4* OR HLT_PFHT330PT30_QuadPFPuppiJet_75_60_45_40_TriplePFPuppiBTagDeepFlavour_2p4* OR HLT_PFHT200PT30_QuadPFPuppiJet_70_40_30_30_TriplePFPuppiBTagDeepFlavour_2p4* OR HLT_DoublePFPuppiJets128_DoublePFPuppiBTagDeepFlavour_2p4*")

phase2_common.toModify(hltBtagTriggerSelection,
                       triggerConditions = triggerConditions_phase2)

HLTPathNames_phase2 = cms.vstring(
    'HLT_DoublePFPuppiJets128_DoublePFPuppiBTagDeepCSV_2p4',
    'HLT_PFHT330PT30_QuadPFPuppiJet_75_60_45_40_TriplePFPuppiBTagDeepFlavour_2p4',
    'HLT_PFHT200PT30_QuadPFPuppiJet_70_40_30_30_TriplePFPuppiBTagDeepFlavour_2p4',
    'HLT_DoublePFPuppiJets128_DoublePFPuppiBTagDeepFlavour_2p4'
)

phase2_common.toModify(hltbTagValidation,
                       HLTPathNames = HLTPathNames_phase2)

phase2_common.toModify(HltVertexValidationVertices,
                       HLTPathNames = HLTPathNames_phase2)

phase2_common.toModify(
    HltVertexValidationVertices,
    Vertex = cms.VInputTag(
            cms.InputTag("hltOfflinePrimaryVertices","","HLT"),
    )
)

phase2_common.toModify(
    hltbTagValidation,
    isPhase2 = cms.bool(True),
    L1Seeds = cms.VPSet(
        cms.PSet(seeds = cms.vstring("pDoublePuppiJet112_112")),
        cms.PSet(seeds = cms.vstring("pPuppiHT400", "pQuadJet70_55_40_40")),
        cms.PSet(seeds = cms.vstring("pPuppiHT400", "pQuadJet70_55_40_40")),
        cms.PSet(seeds = cms.vstring("pDoublePuppiJet112_112"))
    ),
    JetTag = cms.VInputTag(
        cms.InputTag("hltDeepCombinedSecondaryVertexBJetTagsPFPuppiModEta2p4","probb"),
        cms.InputTag("hltPfDeepFlavourJetTagsModEta2p4","probb"),
        cms.InputTag("hltPfDeepFlavourJetTagsModEta2p4","probb"),
        cms.InputTag("hltPfDeepFlavourJetTagsModEta2p4","probb")
    ),
    PathFilters = cms.VPSet(
        cms.PSet(filters = cms.vstring(
            "hltDoublePFPuppiJets128MaxEta2p4",
            "hltDoublePFPuppiJets128Eta2p4MaxDeta1p6",
            "hltBTagPFPuppiDeepCSV0p865DoubleEta2p4"
        )),
        cms.PSet(filters = cms.vstring(
            "hltPFPuppiCentralJetQuad30MaxEta2p4",
            "hlt1PFPuppiCentralJet75MaxEta2p4",
            "hlt2PFPuppiCentralJet60MaxEta2p4",
            "hlt3PFPuppiCentralJet45MaxEta2p4",
            "hlt4PFPuppiCentralJet40MaxEta2p4",
            "hltPFPuppiCentralJetsQuad30HT330MaxEta2p4",
            "hltBTagPFPuppiDeepFlavour0p275Eta2p4TripleEta2p4"
        )),
        cms.PSet(filters = cms.vstring(
            "hltPFPuppiCentralJetQuad30MaxEta2p4",
            "hlt1PFPuppiCentralJet70MaxEta2p4",
            "hlt2PFPuppiCentralJet40MaxEta2p4",
            "hltPFPuppiCentralJetsQuad30HT200MaxEta2p4",
            "hltBTagPFPuppiDeepFlavour0p375Eta2p4TripleEta2p4"
        )),
        cms.PSet(filters = cms.vstring(
            "hltDoublePFPuppiJets128MaxEta2p4",
            "hltDoublePFPuppiJets128Eta2p4MaxDeta1p6",
            "hltBTagPFPuppiDeepFlavour0p935DoubleEta2p4"
        ))
    )
)
