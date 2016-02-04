import FWCore.ParameterSet.Config as cms

allSuperClustersGenParticlesLeptonMatch = cms.EDFilter("MCTruthDeltaRMatcher",
    src = cms.InputTag("allSuperClusters"),
    distMin = cms.double(0.15),
    matchPDGId = cms.vint32(11),
    matched = cms.InputTag("genParticleCandidates")
)


