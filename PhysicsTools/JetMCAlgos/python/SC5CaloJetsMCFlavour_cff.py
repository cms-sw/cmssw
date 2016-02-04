import FWCore.ParameterSet.Config as cms

# Flavour byReference
SC5byRef = cms.EDProducer("JetPartonMatcher",
    jets = cms.InputTag("sisCone5CaloJets"),
    coneSizeToAssociate = cms.double(0.3),
    partons = cms.InputTag("myPartons")
)

# Flavour byValue PhysDef
SC5byValPhys = cms.EDProducer("JetFlavourIdentifier",
    srcByReference = cms.InputTag("SC5byRef"),
    physicsDefinition = cms.bool(True),
    leptonInfo = cms.bool(True)
)

# Flavour byValue AlgoDef
SC5byValAlgo = cms.EDProducer("JetFlavourIdentifier",
    srcByReference = cms.InputTag("SC5byRef"),
    physicsDefinition = cms.bool(False),
    leptonInfo = cms.bool(True)
)
