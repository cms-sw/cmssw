import FWCore.ParameterSet.Config as cms
from HLTriggerOffline.Btag.hltBtagJetMCTools_cff import *

#denominator trigger
hltBtagTriggerSelection = cms.EDFilter( "TriggerResultsFilter",
    triggerConditions = cms.vstring(
      "HLT_PFMET120_PFMHT120_IDTight_v* OR HLT_PFHT330PT30_QuadPFJet_75_60_45_40_v* OR HLT_PFHT380_SixPFJet32_DoublePFBTagCSV_* OR HLT_PFHT380_SixPFJet32_DoublePFBTagDeepCSV_* OR HLT_IsoMu24_eta2p1_v*"),
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
	'HLT_PFHT400_SixPFJet32_DoublePFBTagDeepCSV_',
	'HLT_PFHT400_SixPFJet32_DoublePFBTagDeepJet_',
	'HLT_IsoMu24_eta2p1_v',
	'HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_PFDiJet30_PFBtagDeepCSV_1p5',
	'HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_PFDiJet30_PFBtagDeepJet_1p5',
	#'HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_CaloDiJet30_CaloBtagDeepCSV_1p5',
	'HLT_BTagMu_AK4DiJet20_Mu5_v',
	),
	Vertex = cms.VInputTag(
		cms.InputTag("hltVerticesL3"), 
		cms.InputTag("hltFastPrimaryVertex"), 
		cms.InputTag("hltFastPVPixelVertices"),
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
	'HLT_PFHT330PT30_QuadPFJet_75_60_45_40_v',
	'HLT_PFHT400_SixPFJet32_DoublePFBTagDeepCSV_',
	'HLT_PFHT400_SixPFJet32_DoublePFBTagDeepCSV_',
	'HLT_PFHT400_SixPFJet32_DoublePFBTagDeepJet_',  
	'HLT_IsoMu24_eta2p1_v',
	'HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_PFDiJet30_PFBtagDeepCSV_1p5',
	'HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_PFDiJet30_PFBtagDeepJet_1p5',
	#'HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_CaloDiJet30_CaloBtagDeepCSV_1p5',
	'HLT_BTagMu_AK4DiJet20_Mu5_v',
	),
	JetTag = cms.VInputTag(
		cms.InputTag("hltDeepCombinedSecondaryVertexBJetTagsCalo", "probb"),
		cms.InputTag("hltDeepCombinedSecondaryVertexBJetTagsCalo", "probb"),
		cms.InputTag("hltDeepCombinedSecondaryVertexBJetTagsPF", "probb"),
		cms.InputTag("hltDeepCombinedSecondaryVertexBJetTagsCalo", "probb"),
		cms.InputTag("hltDeepCombinedSecondaryVertexBJetTagsPF", "probb"),
		cms.InputTag("hltDeepCombinedSecondaryVertexBJetTagsPF", "probb"),
		cms.InputTag("hltDeepCombinedSecondaryVertexBJetTagsPF", "probb"),
		cms.InputTag("hltDeepCombinedSecondaryVertexBJetTagsPF", "probb"),
		cms.InputTag("hltDeepCombinedSecondaryVertexBJetTagsPF", "probb"),
		#cms.InputTag("hltDeepCombinedSecondaryVertexBJetTagsCalo", "probb"),
		cms.InputTag("hltBSoftMuonDiJet20L1FastJetL25Jets"),
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

