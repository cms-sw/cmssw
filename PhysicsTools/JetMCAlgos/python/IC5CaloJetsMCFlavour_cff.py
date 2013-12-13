import FWCore.ParameterSet.Config as cms

# Flavour byReference
AK4byRef = cms.EDProducer("JetPartonMatcher",
    jets = cms.InputTag("ak4CaloJets"),
    coneSizeToAssociate = cms.double(0.3),
    partons = cms.InputTag("myPartons")
)

# Flavour byValue PhysDef
AK4byValPhys = cms.EDProducer("JetFlavourIdentifier",
    srcByReference = cms.InputTag("AK4byRef"),
    physicsDefinition = cms.bool(True),
    leptonInfo = cms.bool(True)
)

# Flavour byValue AlgoDef
AK4byValAlgo = cms.EDProducer("JetFlavourIdentifier",
    srcByReference = cms.InputTag("AK4byRef"),
    physicsDefinition = cms.bool(False),
    leptonInfo = cms.bool(True)
)
