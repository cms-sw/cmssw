import FWCore.ParameterSet.Config as cms
from HLTriggerOffline.Btag.HltBtagValidation_cff import *

#define HltVertexValidationVerticesFastSim for the vertex DQM validation (no hltFastPrimaryVertex)
HltVertexValidationVerticesFastSim= cms.EDAnalyzer("HLTVertexPerformanceAnalyzer",
	TriggerResults = cms.InputTag('TriggerResults','',"HLT"),
	HLTPathNames =cms.vstring(
		'HLT_PFMET120_NoiseCleaned_BTagCSV07_v1', 
	#	'HLT_PFMET120_NoiseCleaned_BTagCSV07_v1', 
		'HLT_PFMET120_NoiseCleaned_BTagCSV07_v1'
		),
	Vertex = cms.VInputTag(
		cms.InputTag("hltVerticesL3"), 
#		cms.InputTag("hltFastPrimaryVertex"), 
		cms.InputTag("hltFastPVPixelVertices"),
	)
)

#put all in a path
hltbtagValidationSequenceFastSim = cms.Sequence(
	hltBtagTriggerSelection +
	hltBtagJetMCTools +
	HltVertexValidationVerticesFastSim +
	hltbTagValidation
)
