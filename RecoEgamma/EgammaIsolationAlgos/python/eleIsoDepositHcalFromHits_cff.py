import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaIsolationAlgos.eleHcalExtractorBlocks_cff import *

eleIsoDepositHcalFromHits = cms.EDProducer("CandIsoDepositProducer",
    src = cms.InputTag("pixelMatchGsfElectrons"),
    trackType = cms.string('candidate'),
    MultipleDepositsFlag = cms.bool(False),
    ExtractorPSet = cms.PSet(EleIsoHcalFromHitsExtractorBlock)
)



