import FWCore.ParameterSet.Config as cms

# Flavour byReference
AK5byRef = cms.EDProducer("JetPartonMatcher",
    jets = cms.InputTag("ak5CaloJets"),
    coneSizeToAssociate = cms.double(0.3),
    partons = cms.InputTag("myPartons")
)

# Flavour byValue PhysDef
AK5byValPhys = cms.EDProducer("JetFlavourIdentifier",
    srcByReference = cms.InputTag("AK5byRef"),
    physicsDefinition = cms.bool(True),
    leptonInfo = cms.bool(True)
)

# Flavour byValue AlgoDef
AK5byValAlgo = cms.EDProducer("JetFlavourIdentifier",
    srcByReference = cms.InputTag("AK5byRef"),
    physicsDefinition = cms.bool(False),
    leptonInfo = cms.bool(True)
)
