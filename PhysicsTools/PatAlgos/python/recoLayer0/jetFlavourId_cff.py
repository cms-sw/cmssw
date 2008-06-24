import FWCore.ParameterSet.Config as cms

jetPartons = cms.EDFilter("PartonSelector",
    withLeptons = cms.bool(False)
)

jetPartonAssociation = cms.EDFilter("JetPartonMatcher",
    jets    = cms.InputTag("allLayer0Jets"),
    partons = cms.InputTag("jetPartons"),
    coneSizeToAssociate = cms.double(0.3),
)

jetFlavourAssociation = cms.EDFilter("JetFlavourIdentifier",
    srcByReference    = cms.InputTag("jetPartonAssociation"),
    physicsDefinition = cms.bool(False)
)

# default PAT sequence for jet flavour identification
patJetFlavourId = cms.Sequence(jetPartons * jetPartonAssociation * jetFlavourAssociation)

