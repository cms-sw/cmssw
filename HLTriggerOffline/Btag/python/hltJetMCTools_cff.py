# taken from http://cmssw.cvs.cern.ch/cgi-bin/cmssw.cgi/CMSSW/HLTriggerOffline/BJet/python/hltJetMCTools_cff.py?revision=1.4&view=markup

#define hltJetMCTools for jet/parton matching

import FWCore.ParameterSet.Config as cms

hltPartons = cms.EDProducer("PartonSelector",
   src = cms.InputTag("genParticles"),
    withLeptons = cms.bool(False)
)

hltJetsbyRef = cms.EDProducer("JetPartonMatcher",
    jets = cms.InputTag("hltCaloJetL1FastJetCorrected","","HLT"),
    coneSizeToAssociate = cms.double(0.3),
    partons = cms.InputTag("hltPartons")
)

hltJetsbyValAlgo = cms.EDProducer("JetFlavourIdentifier",
    srcByReference = cms.InputTag("hltJetsbyRef"),
    physicsDefinition = cms.bool(False)
)

hltJetMCTools = cms.Sequence(hltPartons*hltJetsbyRef*hltJetsbyValAlgo)
