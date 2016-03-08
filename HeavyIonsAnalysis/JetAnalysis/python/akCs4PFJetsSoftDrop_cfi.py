import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.PFJetParameters_cfi import *
from RecoJets.JetProducers.AnomalousCellParameters_cfi import *
#from RecoJets.JetProducers.ak4PFJets_cfi import ak4PFJets

akCs4PFJetsSoftDrop = cms.EDProducer(
		        "FastjetJetProducer",
		        PFJetParameters,
		        AnomalousCellParameters,
			jetAlgorithm = cms.string("AntiKt"),
			rParam       = cms.double(0.4),
    useSoftDrop = cms.bool(True),
    zcut = cms.double(0.1),
    beta = cms.double(0.0),
    R0   = cms.double(0.4),
    useExplicitGhosts = cms.bool(True),
    writeCompound = cms.bool(True),
    jetCollInstanceName=cms.string("SubJets")
    )
akCs4PFJetsSoftDrop.src    = cms.InputTag("akCs4PFJets","pfParticlesCs")
