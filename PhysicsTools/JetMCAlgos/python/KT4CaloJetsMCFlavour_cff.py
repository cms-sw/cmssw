import FWCore.ParameterSet.Config as cms

# Flavour byReference
KT4byRef = cms.EDFilter("JetPartonMatcher",
    jets = cms.InputTag("kt4CaloJets"),
    coneSizeToAssociate = cms.double(0.3),
    partons = cms.InputTag("myPartons")
)

# Flavour byValue PhysDef
KT4byValPhys = cms.EDFilter("JetFlavourIdentifier",
    srcByReference = cms.InputTag("KT4byRef"),
    physicsDefinition = cms.bool(True)
)

# Flavour byValue AlgoDef
KT4byValAlgo = cms.EDFilter("JetFlavourIdentifier",
    srcByReference = cms.InputTag("KT4byRef"),
    physicsDefinition = cms.bool(False)
)


