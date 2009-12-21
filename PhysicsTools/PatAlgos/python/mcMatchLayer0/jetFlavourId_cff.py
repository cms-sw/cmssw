import FWCore.ParameterSet.Config as cms

patAK5CaloJetPartons = cms.EDFilter("PartonSelector",
    withLeptons = cms.bool(False),
    src = cms.InputTag("genParticles")                            
)

patAK5CaloJetPartonAssociation = cms.EDFilter("JetPartonMatcher",
    jets    = cms.InputTag("ak5CaloJets"),
    partons = cms.InputTag("patAK5CaloJetPartons"),
    coneSizeToAssociate = cms.double(0.3),
)

patAK5CaloJetFlavourAssociation = cms.EDFilter("JetFlavourIdentifier",
    srcByReference    = cms.InputTag("patAK5CaloJetPartonAssociation"),
    physicsDefinition = cms.bool(False)
)

# default PAT sequence for jet flavour identification
patJetFlavourId = cms.Sequence(patAK5CaloJetPartons * patAK5CaloJetPartonAssociation * patAK5CaloJetFlavourAssociation)

