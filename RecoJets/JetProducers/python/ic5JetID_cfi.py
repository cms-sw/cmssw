import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.JetIDParams_cfi import *

ic5JetID = cms.EDProducer('JetIDProducer', JetIDParams,
        src = cms.InputTag('iterativeCone5CaloJets')
        
)
