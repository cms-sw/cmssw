import FWCore.ParameterSet.Config as cms

# Flavour byReference
AK8byRef = cms.EDProducer("JetPartonMatcher",
    jets = cms.InputTag("ak8CaloJets"),
    coneSizeToAssociate = cms.double(0.3),
    partons = cms.InputTag("myPartons")
)

# Flavour byValue PhysDef
AK8byValPhys = cms.EDProducer("JetFlavourIdentifier",
    srcByReference = cms.InputTag("AK8byRef"),
    physicsDefinition = cms.bool(True),
    leptonInfo = cms.bool(True)
)

# Flavour byValue AlgoDef
AK8byValAlgo = cms.EDProducer("JetFlavourIdentifier",
    srcByReference = cms.InputTag("AK8byRef"),
    physicsDefinition = cms.bool(False),
    leptonInfo = cms.bool(True)
)
