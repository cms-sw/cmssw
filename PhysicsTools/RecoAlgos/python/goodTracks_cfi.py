import FWCore.ParameterSet.Config as cms

goodTracks = cms.EDProducer("TrackViewCandidateProducer",
    src = cms.InputTag("generalTracks"),
    particleType = cms.string('mu+'),
    cut = cms.string('pt > 0')
)


# foo bar baz
# R6DdbsX5PbBrm
# W8l92abrZhxAH
