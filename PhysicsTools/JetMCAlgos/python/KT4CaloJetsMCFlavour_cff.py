import FWCore.ParameterSet.Config as cms

# Flavour byReference
KT4byRef = cms.EDProducer("JetPartonMatcher",
    jets = cms.InputTag("kt4CaloJets"),
    coneSizeToAssociate = cms.double(0.3),
    partons = cms.InputTag("myPartons")
)

# Flavour byValue PhysDef
KT4byValPhys = cms.EDProducer("JetFlavourIdentifier",
    srcByReference = cms.InputTag("KT4byRef"),
    physicsDefinition = cms.bool(True),
    leptonInfo = cms.bool(True)
)

# Flavour byValue AlgoDef
KT4byValAlgo = cms.EDProducer("JetFlavourIdentifier",
    srcByReference = cms.InputTag("KT4byRef"),
    physicsDefinition = cms.bool(False),
    leptonInfo = cms.bool(True)
)
