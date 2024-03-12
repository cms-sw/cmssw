import FWCore.ParameterSet.Config as cms

goodStandAloneMuonTracks = cms.EDProducer("TrackViewCandidateProducer",
    src = cms.InputTag("standAloneMuons","UpdatedAtVtx"),
    particleType = cms.string('mu+'),
    cut = cms.string('pt > 0')
)


# foo bar baz
# 849dg6ivg3T1e
# YqUopzIO3WOUV
