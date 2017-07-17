import FWCore.ParameterSet.Config as cms

l1RecoTree = cms.EDAnalyzer("L1RecoTreeProducer",
  vtxToken                = cms.untracked.InputTag("offlinePrimaryVertices"),
  maxVtx                  = cms.uint32(100)
)

