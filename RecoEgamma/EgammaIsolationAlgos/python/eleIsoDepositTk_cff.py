import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaIsolationAlgos.eleTrackExtractorBlocks_cff import *

eleIsoDepositTk = cms.EDProducer("CandIsoDepositProducer",
    src = cms.InputTag("gedGsfElectrons"),
    trackType = cms.string('candidate'),
    MultipleDepositsFlag = cms.bool(False),
    ExtractorPSet = cms.PSet(EleIsoTrackExtractorBlock)
)



