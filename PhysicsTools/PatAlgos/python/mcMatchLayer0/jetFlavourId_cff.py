import FWCore.ParameterSet.Config as cms

jetPartons = cms.EDFilter("PartonSelector",
    withLeptons = cms.bool(False),
    src = cms.InputTag("genParticles")                            
)

jetPartonAssociation = cms.EDFilter("JetPartonMatcher",
    jets    = cms.InputTag("ak5CaloJets"),
    partons = cms.InputTag("jetPartons"),
    coneSizeToAssociate = cms.double(0.3),
)

jetFlavourAssociation = cms.EDFilter("JetFlavourIdentifier",
    srcByReference    = cms.InputTag("jetPartonAssociation"),
    physicsDefinition = cms.bool(False)
)

# default PAT sequence for jet flavour identification
jetFlavourId = cms.Sequence(jetPartons * jetPartonAssociation * jetFlavourAssociation)

