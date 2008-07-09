import FWCore.ParameterSet.Config as cms

jetPartons = cms.EDFilter("PartonSelector",
    withLeptons = cms.bool(False)
)

jetPartonAssociation = cms.EDFilter("JetPartonMatcher",
    jets = cms.InputTag("allLayer0CaloJets"),
    coneSizeToAssociate = cms.double(0.3),
    partons = cms.InputTag("jetPartons")
)

jetFlavourAssociation = cms.EDFilter("JetFlavourIdentifier",
    srcByReference = cms.InputTag("jetPartonAssociation"),
    physicsDefinition = cms.bool(False)
)

patJetFlavourId = cms.Sequence(jetPartons*jetPartonAssociation*jetFlavourAssociation)

