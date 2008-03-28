import FWCore.ParameterSet.Config as cms

goodTracks = cms.EDProducer("TrackViewCandidateProducer",
    src = cms.InputTag("ctfWithMaterialTracks"),
    particleType = cms.string('mu+'),
    cut = cms.string('pt > 0')
)


