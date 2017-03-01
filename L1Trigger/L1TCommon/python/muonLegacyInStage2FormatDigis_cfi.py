import FWCore.ParameterSet.Config as cms

import os

muonLegacyInStage2FormatDigis = cms.EDProducer('L1TMuonLegacyConverter',
    muonSource = cms.InputTag("gtDigis","")
)



