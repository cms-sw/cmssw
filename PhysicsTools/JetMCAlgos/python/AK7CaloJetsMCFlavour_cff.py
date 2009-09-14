import FWCore.ParameterSet.Config as cms

# Flavour byReference
AK7byRef = cms.EDFilter("JetPartonMatcher",
    jets = cms.InputTag("ak7CaloJets"),
    coneSizeToAssociate = cms.double(0.3),
    partons = cms.InputTag("myPartons")
)

# Flavour byValue PhysDef
AK7byValPhys = cms.EDFilter("JetFlavourIdentifier",
    srcByReference = cms.InputTag("AK7byRef"),
    physicsDefinition = cms.bool(True)
)

# Flavour byValue AlgoDef
AK7byValAlgo = cms.EDFilter("JetFlavourIdentifier",
    srcByReference = cms.InputTag("AK7byRef"),
    physicsDefinition = cms.bool(False)
)


