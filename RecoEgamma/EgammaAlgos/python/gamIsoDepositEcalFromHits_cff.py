import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaAlgos.gamEcalExtractorBlocks_cff import *

gamIsoDepositEcalFromHits = cms.EDProducer("CandIsoDepositProducer",
    src = cms.InputTag("photons"),
    trackType = cms.string('candidate'),
    MultipleDepositsFlag = cms.bool(False),
    ExtractorPSet = cms.PSet(GamIsoEcalFromHitsExtractorBlock)
)



