#define hltJetMCTools for jet/parton matching

import FWCore.ParameterSet.Config as cms

hltBtagPartons = cms.EDProducer("PartonSelector",
   src = cms.InputTag("genParticles"),
    withLeptons = cms.bool(False)
)

hltBtagJetsbyRef = cms.EDProducer("JetPartonMatcher",
    jets = cms.InputTag("hltCaloJetL1FastJetCorrected","","HLT"),
    coneSizeToAssociate = cms.double(0.3),
    partons = cms.InputTag("hltPartons")
)

hltBtagJetsbyValAlgo = cms.EDProducer("JetFlavourIdentifier",
    srcByReference = cms.InputTag("hltJetsbyRef"),
    physicsDefinition = cms.bool(False)
)

hltBtagJetMCTools = cms.Sequence(hltBtagPartons*hltBtagJetsbyRef*hltBtagJetsbyValAlgo)
