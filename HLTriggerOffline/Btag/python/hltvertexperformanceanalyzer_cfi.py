#define VertexValidationVertices for the vertex DQM validation
process.VertexValidationVertices= cms.EDAnalyzer("HLTVertexPerformanceAnalyzer",
TriggerResults = cms.InputTag('TriggerResults','',fileini.processname),
HLTPathNames = cms.vstring(fileini.vertex_pathes),
Vertex = fileini.vertex_modules,
)
