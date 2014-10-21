import FWCore.ParameterSet.Config as cms
from HLTriggerOffline.Btag.hltBtagJetMCTools_cff import *
from HLTriggerOffline.Btag.HltBtagValidation_cff import *

#denominator trigger
#hltBtagTriggerSelection = cms.EDFilter( "TriggerResultsFilter",
#    triggerConditions = cms.vstring(
#      "HLT_PFJet40*"),
#    hltResults = cms.InputTag( "TriggerResults", "", "HLT" ),
#    l1tResults = cms.InputTag( "simGtDigis" ),
##    l1tResults = cms.InputTag( "hltGtDigis" ),
#    throw = cms.bool( True )
#)

#define HltVertexValidationVerticesFastSim for the vertex DQM validation (no hltFastPrimaryVertex)
HltVertexValidationVerticesFastSim= cms.EDAnalyzer("HLTVertexPerformanceAnalyzer",
TriggerResults = cms.InputTag('TriggerResults','',"HLT"),
HLTPathNames =cms.vstring(
	'HLT_BTagCSV07_v1', 
#	'HLT_BTagCSV07_v1', 
	'HLT_BTagCSV07_v1'
	),
	Vertex = cms.VInputTag(
		cms.InputTag("hltVerticesL3"), 
#		cms.InputTag("hltFastPrimaryVertex"), 
		cms.InputTag("hltFastPVPixelVertices"),
	)
)

#put all in a path
hltbtagValidationSequenceFastSim = cms.Sequence(
#hltBtagTriggerSelection +
	hltBtagJetMCTools +
	HltVertexValidationVerticesFastSim +
	hltbTagValidation
)
