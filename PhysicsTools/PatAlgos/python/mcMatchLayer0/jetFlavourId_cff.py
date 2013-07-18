import FWCore.ParameterSet.Config as cms

patJetPartons = cms.EDProducer("PartonSelector",
    withLeptons = cms.bool(False),
    src = cms.InputTag("genParticles")                            
)

patJetPartonAssociation = cms.EDProducer("JetPartonMatcher",
    jets    = cms.InputTag("ak5CaloJets"),
    partons = cms.InputTag("patJetPartons"),
    coneSizeToAssociate = cms.double(0.3),
)

patJetFlavourAssociation = cms.EDProducer("JetFlavourIdentifier",
    srcByReference    = cms.InputTag("patJetPartonAssociation"),
    physicsDefinition = cms.bool(False)
)

# default PAT sequence for jet flavour identification
patJetFlavourId = cms.Sequence(patJetPartons * patJetPartonAssociation * patJetFlavourAssociation)

