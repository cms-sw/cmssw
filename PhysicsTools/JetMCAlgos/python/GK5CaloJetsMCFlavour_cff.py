import FWCore.ParameterSet.Config as cms

# Flavour byReference
GK5byRef = cms.EDProducer("JetPartonMatcher",
    jets = cms.InputTag("gk5CaloJets"),
    coneSizeToAssociate = cms.double(0.3),
    partons = cms.InputTag("myPartons")
)

# Flavour byValue PhysDef
GK5byValPhys = cms.EDProducer("JetFlavourIdentifier",
    srcByReference = cms.InputTag("GK5byRef"),
    physicsDefinition = cms.bool(True),
    leptonInfo = cms.bool(True)
)

# Flavour byValue AlgoDef
GK5byValAlgo = cms.EDProducer("JetFlavourIdentifier",
    srcByReference = cms.InputTag("GK5byRef"),
    physicsDefinition = cms.bool(False),
    leptonInfo = cms.bool(True)
)
