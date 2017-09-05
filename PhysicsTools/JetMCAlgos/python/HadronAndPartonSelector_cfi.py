import FWCore.ParameterSet.Config as cms

# select hadrons and partons for the jet flavour
selectedHadronsAndPartons = cms.EDProducer('HadronAndPartonSelector',
    src = cms.InputTag("generator"),
    particles = cms.InputTag("genParticles"),
    partonMode = cms.string("Auto"),
    fullChainPhysPartons = cms.bool(True)
)
# select hadrons and partons for the slimmedGenJetsFlavourInfos, required for origin identification

selectedHadronsAndPartonsForGenJetsFlavourInfos = selectedHadronsAndPartons.clone(particles = "prunedGenParticles")
