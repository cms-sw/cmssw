import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaIsolationAlgos.eleEcalExtractorBlocks_cff import *

eleIsoDepositEcalFromHits = cms.EDProducer("CandIsoDepositProducer",
    src = cms.InputTag("gsfElectrons"),
    trackType = cms.string('candidate'),
    MultipleDepositsFlag = cms.bool(False),
    ExtractorPSet = cms.PSet(EleIsoEcalFromHitsExtractorBlock)
)



