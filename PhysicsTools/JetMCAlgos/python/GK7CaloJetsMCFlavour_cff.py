import FWCore.ParameterSet.Config as cms

# Flavour byReference
GK7byRef = cms.EDProducer("JetPartonMatcher",
    jets = cms.InputTag("gk7CaloJets"),
    coneSizeToAssociate = cms.double(0.3),
    partons = cms.InputTag("myPartons")
)

# Flavour byValue PhysDef
GK7byValPhys = cms.EDProducer("JetFlavourIdentifier",
    srcByReference = cms.InputTag("GK7byRef"),
    physicsDefinition = cms.bool(True),
    leptonInfo = cms.bool(True)
)

# Flavour byValue AlgoDef
GK7byValAlgo = cms.EDProducer("JetFlavourIdentifier",
    srcByReference = cms.InputTag("GK7byRef"),
    physicsDefinition = cms.bool(False),
    leptonInfo = cms.bool(True)
)
