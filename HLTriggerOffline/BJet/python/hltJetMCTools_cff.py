import FWCore.ParameterSet.Config as cms

# hlt jet mc tools
# only run on events with L2 jets
require_iterativeCone5CaloJets = cms.EDFilter("RequireModule",
    requirement = cms.InputTag("iterativeCone5CaloJets","","HLT")
)

# see "PhysicsTools/JetMCAlgos/data/SelectPartons.cff"
hltPartons = cms.EDFilter("PartonSelector",
    withLeptons = cms.bool(False)
)

# flavour by reference
# see "PhysicsTools/JetMCAlgos/data/IC5CaloJetsMCFlavour.cff"
hltIC5byRef = cms.EDFilter("JetPartonMatcher",
    jets = cms.InputTag("iterativeCone5CaloJets","","HLT"),
    coneSizeToAssociate = cms.double(0.3),
    partons = cms.InputTag("hltPartons")
)

# flavour by value, physics definition
# see "PhysicsTools/JetMCAlgos/data/IC5CaloJetsMCFlavour.cff"
hltIC5byValPhys = cms.EDFilter("JetFlavourIdentifier",
    srcByReference = cms.InputTag("hltIC5byRef"),
    physicsDefinition = cms.bool(True)
)

# flavour by value, algorithmic definition
# see "PhysicsTools/JetMCAlgos/data/IC5CaloJetsMCFlavour.cff"
hltIC5byValAlgo = cms.EDFilter("JetFlavourIdentifier",
    srcByReference = cms.InputTag("hltIC5byRef"),
    physicsDefinition = cms.bool(False)
)

hltJetMCTools = cms.Sequence(require_iterativeCone5CaloJets*hltPartons*hltIC5byRef*hltIC5byValPhys+hltIC5byValAlgo)

