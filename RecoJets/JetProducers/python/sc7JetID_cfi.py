import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.JetIDParams_cfi import *

sc7JetID = cms.EDProducer('JetIDProducer', JetIDParams,
        src = cms.InputTag('sisCone7CaloJets')
        
)
