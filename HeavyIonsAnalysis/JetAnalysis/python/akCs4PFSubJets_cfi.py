import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.PFJetParameters_cfi import *
from RecoJets.JetProducers.AnomalousCellParameters_cfi import *
from RecoJets.JetProducers.SubJetParameters_cfi import SubJetParameters

akCs4PFSubJets = cms.EDProducer(
    "SubJetProducer",
    PFJetParameters,
    AnomalousCellParameters,
    SubJetParameters,
    jetAlgorithm = cms.string("CambridgeAachen"),
    rParam       = cms.double(0.4),
    nSubjets     = cms.int32(2),
    useSoftDrop  = cms.bool(False),
    #zcut = cms.double(0.1),
    #useExplicitGhosts = cms.bool(True),
    #writeCompound = cms.bool(True),
    jetCollInstanceName=cms.string("SubJets")
    )
akCs4PFSubJets.src    = cms.InputTag("akCs4PFJets","pfParticlesCs")
