import FWCore.ParameterSet.Config as cms

# Flavour byReference
SC7byRef = cms.EDProducer("JetPartonMatcher",
    jets = cms.InputTag("sisCone7CaloJets"),
    coneSizeToAssociate = cms.double(0.3),
    partons = cms.InputTag("myPartons")
)

# Flavour byValue PhysDef
SC7byValPhys = cms.EDProducer("JetFlavourIdentifier",
    srcByReference = cms.InputTag("SC7byRef"),
    physicsDefinition = cms.bool(True),
    leptonInfo = cms.bool(True)
)

# Flavour byValue AlgoDef
SC7byValAlgo = cms.EDProducer("JetFlavourIdentifier",
    srcByReference = cms.InputTag("SC7byRef"),
    physicsDefinition = cms.bool(False),
    leptonInfo = cms.bool(True)
)
