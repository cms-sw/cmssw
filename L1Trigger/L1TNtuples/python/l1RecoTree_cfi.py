import FWCore.ParameterSet.Config as cms

l1RecoTree = cms.EDAnalyzer("L1RecoTreeProducer",
  vtxToken                = cms.untracked.InputTag("offlinePrimaryVertices"),
  maxVtx                  = cms.uint32(100)
)

# foo bar baz
# RBlXp8Lf1fEP4
