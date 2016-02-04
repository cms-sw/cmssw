import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.JetIDParams_cfi import *

ca4JetID = cms.EDProducer('JetIDProducer', JetIDParams,
        src = cms.InputTag('ca4CaloJets')
        
)
