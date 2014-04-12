import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.JetIDParams_cfi import *

sc5JetID = cms.EDProducer('JetIDProducer', JetIDParams,
        src = cms.InputTag('sisCone5CaloJets')
        
)
