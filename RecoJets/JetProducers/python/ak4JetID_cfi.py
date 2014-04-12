import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.JetIDParams_cfi import *

ak4JetID = cms.EDProducer('JetIDProducer', JetIDParams,
        src = cms.InputTag('ak4CaloJets')
        
)
