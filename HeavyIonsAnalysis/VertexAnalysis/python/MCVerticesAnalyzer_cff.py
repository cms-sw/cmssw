import FWCore.ParameterSet.Config as cms

pileup = cms.EDAnalyzer(
    "MCVerticesAnalyzer",
    verbose                      = cms.untracked.int32(0),
    dumpAllEvents                = cms.untracked.int32(0),
    vertexCollLabel              = cms.untracked.InputTag('offlinePrimaryVertices')
)
