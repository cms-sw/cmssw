import FWCore.ParameterSet.Config as cms

import os

simGmtCaloSumDigis = cms.EDProducer('L1TMuonCaloSumProducer',
    caloStage2Layer2Label = cms.InputTag("simCaloStage2Layer1Digis"),
)
