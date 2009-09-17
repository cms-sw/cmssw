import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.JetIDParams_cfi import *

gk5JetID = cms.EDProducer('JetIDProducer', JetIDParams,
        src = cms.InputTag('gk5CaloJets')
        
)
