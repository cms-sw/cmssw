import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.JetIDParams_cfi import *

ic7JetID = cms.EDProducer('JetIDProducer', JetIDParams,
        src = cms.InputTag('iterativeCone7CaloJets')
        
)
