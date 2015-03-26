#define hltBtagJetMCTools for jet/parton matching

import FWCore.ParameterSet.Config as cms

hltBtagPartons = cms.EDProducer("PartonSelector",
   src = cms.InputTag("genParticles"),
    withLeptons = cms.bool(False)
)

hltBtagJetsbyRef = cms.EDProducer("JetPartonMatcher",
    jets = cms.InputTag("hltBtagCaloJetL1FastJetCorrected","","HLT"),
    coneSizeToAssociate = cms.double(0.3),
    partons = cms.InputTag("hltBtagPartons")
)

hltBtagJetsbyValAlgo = cms.EDProducer("JetFlavourIdentifier",
    srcByReference = cms.InputTag("hltBtagJetsbyRef"),
    physicsDefinition = cms.bool(False)
)

hltBtagJetMCTools = cms.Sequence(hltBtagPartons*hltBtagJetsbyRef*hltBtagJetsbyValAlgo)
