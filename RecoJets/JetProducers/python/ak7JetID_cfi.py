import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.JetIDParams_cfi import *

ak7JetID = cms.EDProducer('JetIDProducer', JetIDParams,
        src = cms.InputTag('ak7CaloJets')
        
)
