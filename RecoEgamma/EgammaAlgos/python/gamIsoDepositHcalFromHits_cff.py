import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaAlgos.gamHcalExtractorBlocks_cff import *

gamIsoDepositHcalFromHits = cms.EDProducer("CandIsoDepositProducer",
    src = cms.InputTag("photons"),
    trackType = cms.string('candidate'),
    MultipleDepositsFlag = cms.bool(False),
    ExtractorPSet = cms.PSet(GamIsoHcalFromHitsExtractorBlock)
)



