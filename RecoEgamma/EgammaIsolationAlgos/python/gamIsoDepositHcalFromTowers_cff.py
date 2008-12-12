import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaIsolationAlgos.gamHcalExtractorBlocks_cff import *

gamIsoDepositHcalFromTowers = cms.EDProducer("CandIsoDepositProducer",
    src = cms.InputTag("photons"),
    trackType = cms.string('candidate'),
    MultipleDepositsFlag = cms.bool(False),
    ExtractorPSet = cms.PSet(GamIsoHcalFromTowersExtractorBlock)
)
