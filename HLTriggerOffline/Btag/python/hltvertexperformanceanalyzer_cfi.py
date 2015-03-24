#define VertexValidationVertices for the vertex DQM validation
process.VertexValidationVertices= cms.EDAnalyzer("HLTVertexPerformanceAnalyzer",
        SimVertexCollection = cms.InputTag("g4SimHits"),
	TriggerResults = cms.InputTag('TriggerResults',),
	HLTPathNames = cms.vstring('HLT_PFMHT100_SingleCentralJet60_BTagCSV0p6_v1'),
	Vertex = cms.VInputTag(cms.InputTag("hltFastPrimaryVertex"), cms.InputTag("hltFastPVPixelVertices"), cms.InputTag("hltVerticesL3"))
)



