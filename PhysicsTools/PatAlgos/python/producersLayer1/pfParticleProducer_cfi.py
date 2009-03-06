import FWCore.ParameterSet.Config as cms

allLayer1PFParticles = cms.EDProducer("PATPFParticleProducer",
    # General configurables
    pfCandidateSource = cms.InputTag("topProjection:PFCandidates"),

    # MC matching configurables
    addGenMatch = cms.bool(False),
    embedGenMatch = cms.bool(False),
    # what is this ?
    # genParticleMatch = cms.InputTag(""), ## particles source to be used for the matching

)


