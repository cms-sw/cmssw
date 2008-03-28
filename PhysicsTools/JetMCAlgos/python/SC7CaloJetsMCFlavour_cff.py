import FWCore.ParameterSet.Config as cms

# Flavour byReference
SC7byRef = cms.EDFilter("JetPartonMatcher",
    jets = cms.InputTag("sisCone7CaloJets"),
    coneSizeToAssociate = cms.double(0.3),
    partons = cms.InputTag("myPartons")
)

# Flavour byValue PhysDef
SC7byValPhys = cms.EDFilter("JetFlavourIdentifier",
    srcByReference = cms.InputTag("SC7byRef"),
    physicsDefinition = cms.bool(True)
)

# Flavour byValue AlgoDef
SC7byValAlgo = cms.EDFilter("JetFlavourIdentifier",
    srcByReference = cms.InputTag("SC7byRef"),
    physicsDefinition = cms.bool(False)
)


