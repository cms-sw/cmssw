import FWCore.ParameterSet.Config as cms

# Flavour byReference
KT6byRef = cms.EDFilter("JetPartonMatcher",
    jets = cms.InputTag("kt6CaloJets"),
    coneSizeToAssociate = cms.double(0.3),
    partons = cms.InputTag("myPartons")
)

# Flavour byValue PhysDef
KT6byValPhys = cms.EDFilter("JetFlavourIdentifier",
    srcByReference = cms.InputTag("KT6byRef"),
    physicsDefinition = cms.bool(True)
)

# Flavour byValue AlgoDef
KT6byValAlgo = cms.EDFilter("JetFlavourIdentifier",
    srcByReference = cms.InputTag("KT6byRef"),
    physicsDefinition = cms.bool(False)
)


