### This file defines the ExoticaValidationSequence, to be put
### together with the other sequences in
### HLTriggerOffline/Common/python/HLTValidation_cff.py
### Also defines some Producers.

import FWCore.ParameterSet.Config as cms

from HLTriggerOffline.Exotica.hltExoticaValidator_cfi import *

# We create a basic HT variable here
recoExoticaValidationHT = cms.EDProducer(
    "PFMETProducer",
    src = cms.InputTag("ak4PFJetsCHS"),
    alias = cms.string('PFMHT'),
    globalThreshold = cms.double(30.0),
    calculateSignificance = cms.bool(False),
    jets = cms.InputTag("ak4PFJetsCHS") # for significance calculation
    )

recoExoticaValidationMETNoMu = cms.EDProducer( 
    "HLTMhtProducer",                                                                                                                       
    usePt = cms.bool( True ),
    minPtJet = cms.double( 0.0 ),
    jetsLabel = cms.InputTag( "ak4PFJetsCHS" ),
    minNJet = cms.int32( 0 ),
    maxEtaJet = cms.double( 999.0 ),
    excludePFMuons = cms.bool( True ),
    pfCandidatesLabel = cms.InputTag("particleFlow")
    )

recoExoticaValidationMHTNoMu = cms.EDProducer( 
    "HLTHtMhtProducer",                                                                                                                     
    usePt = cms.bool( False ), 
    minPtJetHt = cms.double( 20.0 ),
    maxEtaJetMht = cms.double( 5.2 ),
    minNJetMht = cms.int32( 0 ),
    jetsLabel = cms.InputTag( "ak4PFJetsCHS" ),
    maxEtaJetHt = cms.double( 5.2 ),
    minPtJetMht = cms.double( 20.0 ),
    minNJetHt = cms.int32( 0 ),
    pfCandidatesLabel = cms.InputTag( "particleFlow" ),
    excludePFMuons = cms.bool( True )
    )   

recoExoticaValidationCaloHT = cms.EDProducer(
    "CaloMETProducer",
    src = cms.InputTag("ak4CaloJets"),
    noHF = cms.bool( True ),
    alias = cms.string('CaloMHT'),
    globalThreshold = cms.double(30.0),
    calculateSignificance = cms.bool( False ),
    jets = cms.InputTag("ak4CaloJets") # for significance calculation
    )

ExoticaValidationProdSeq = cms.Sequence(
    recoExoticaValidationHT + recoExoticaValidationMETNoMu + recoExoticaValidationMHTNoMu + recoExoticaValidationCaloHT
    )

ExoticaValidationSequence = cms.Sequence(
    hltExoticaValidator
    )

#HLTExoticaVal_FastSim = cms.Sequence(
#    recoExoticaValidationHLTFastSim_seq +
#    hltExoticaValidator
#    )
