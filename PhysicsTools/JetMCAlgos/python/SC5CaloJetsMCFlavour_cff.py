import FWCore.ParameterSet.Config as cms

# Flavour byReference
SC5byRef = cms.EDFilter("JetPartonMatcher",
    jets = cms.InputTag("sisCone5CaloJets"),
    coneSizeToAssociate = cms.double(0.3),
    partons = cms.InputTag("myPartons")
)

# Flavour byValue PhysDef
SC5byValPhys = cms.EDFilter("JetFlavourIdentifier",
    srcByReference = cms.InputTag("SC5byRef"),
    physicsDefinition = cms.bool(True)
)

# Flavour byValue AlgoDef
SC5byValAlgo = cms.EDFilter("JetFlavourIdentifier",
    srcByReference = cms.InputTag("SC5byRef"),
    physicsDefinition = cms.bool(False)
)


