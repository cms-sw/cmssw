import FWCore.ParameterSet.Config as cms

# Flavour byReference
IC5byRef = cms.EDProducer("JetPartonMatcher",
    jets = cms.InputTag("iterativeCone5CaloJets"),
    coneSizeToAssociate = cms.double(0.3),
    partons = cms.InputTag("myPartons")
)

# Flavour byValue PhysDef
IC5byValPhys = cms.EDProducer("JetFlavourIdentifier",
    srcByReference = cms.InputTag("IC5byRef"),
    physicsDefinition = cms.bool(True),
    leptonInfo = cms.bool(True)
)

# Flavour byValue AlgoDef
IC5byValAlgo = cms.EDProducer("JetFlavourIdentifier",
    srcByReference = cms.InputTag("IC5byRef"),
    physicsDefinition = cms.bool(False),
    leptonInfo = cms.bool(True)
)
