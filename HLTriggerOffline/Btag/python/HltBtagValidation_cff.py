import FWCore.ParameterSet.Config as cms
from HLTriggerOffline.Btag.hltBtagJetMCTools_cff import *

#denominator trigger
hltBtagTriggerSelection = cms.EDFilter( "TriggerResultsFilter",
    triggerConditions = cms.vstring(
      "HLT_PFMET170*"),
    hltResults = cms.InputTag( "TriggerResults", "", "HLT" ),
#    l1tResults = cms.InputTag( "simGtDigis" ),
    l1tResults = cms.InputTag( "gtDigis" ),
    throw = cms.bool( True )
)

#correct the jet used for the matching
hltBtagJetsbyRef.jets = cms.InputTag("hltSelector4CentralJetsL1FastJet")

#define HltVertexValidationVertices for the vertex DQM validation
HltVertexValidationVertices= cms.EDAnalyzer("HLTVertexPerformanceAnalyzer",
	TriggerResults = cms.InputTag('TriggerResults','',"HLT"),
	HLTPathNames =cms.vstring(
	'HLT_PFMET120_NoiseCleaned_BTagCSV07_v1', 
	'HLT_PFMET120_NoiseCleaned_BTagCSV07_v1', 
	'HLT_PFMET120_NoiseCleaned_BTagCSV07_v1'
	),
	Vertex = cms.VInputTag(
		cms.InputTag("hltVerticesL3"), 
		cms.InputTag("hltFastPrimaryVertex"), 
		cms.InputTag("hltFastPVPixelVertices"),
	)
)

#define bTagValidation for the b-tag DQM validation (distribution plot)
hltbTagValidation = cms.EDAnalyzer("HLTBTagPerformanceAnalyzer",
	TriggerResults = cms.InputTag('TriggerResults','','HLT'),
	HLTPathNames = cms.vstring('HLT_PFMET120_NoiseCleaned_BTagCSV07_v1'),
	JetTag = cms.VInputTag(cms.InputTag("hltL3CombinedSecondaryVertexBJetTags")),
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
	hltBtagTriggerSelection +
	hltBtagJetMCTools +
	HltVertexValidationVertices +
	hltbTagValidation
)

