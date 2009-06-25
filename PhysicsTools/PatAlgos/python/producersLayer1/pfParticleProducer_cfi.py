import FWCore.ParameterSet.Config as cms

allLayer1PFParticles = cms.EDProducer("PATPFParticleProducer",
    # General configurables
    pfCandidateSource = cms.InputTag("noJet"),

    # MC matching configurables
    addGenMatch = cms.bool(False),
    genParticleMatch = cms.InputTag(""),   ## particles source to be used for the MC matching
                                           ## must be an InputTag or VInputTag to a product of
                                           ## type edm::Association<reco::GenParticleCollection>
    embedGenMatch = cms.bool(False),       ## embed gen match inside the object instead of storing the ref

)


