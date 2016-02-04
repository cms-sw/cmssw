import FWCore.ParameterSet.Config as cms

allStandAloneMuonTracksGenParticlesLeptonMatch = cms.EDFilter("MCTruthDeltaRMatcher",
    src = cms.InputTag("allStandAloneMuonTracks"),
    # note the unusually large value due to 
    # poor resolution
    distMin = cms.double(0.5),
    matchPDGId = cms.vint32(13),
    matched = cms.InputTag("genParticleCandidates")
)


