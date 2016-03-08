import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.PFJetParameters_cfi import *
from RecoJets.JetProducers.AnomalousCellParameters_cfi import *
#from RecoJets.JetProducers.ak4PFJets_cfi import ak4PFJets

akCs4PFJetsFilter = cms.EDProducer(
		        "FastjetJetProducer",
		        PFJetParameters,
		        AnomalousCellParameters,
			jetAlgorithm = cms.string("AntiKt"),
			rParam       = cms.double(0.4),
    useFiltering = cms.bool(True),
    nFilt = cms.int32(4),
    rFilt = cms.double(0.15),
    useExplicitGhosts = cms.bool(True),
    writeCompound = cms.bool(True),
    jetCollInstanceName=cms.string("SubJets")
    )
akCs4PFJetsFilter.src    = cms.InputTag("akCs4PFJets","pfParticlesCs")
