import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaIsolationAlgos.gamTrackExtractorBlocks_cff import *

gamIsoDepositTk = cms.EDProducer("CandIsoDepositProducer",
    src = cms.InputTag("photons"),
    trackType = cms.string('candidate'),
    MultipleDepositsFlag = cms.bool(False),
    ExtractorPSet = cms.PSet(GamIsoTrackExtractorBlock)
)



# foo bar baz
# ebXsHVt9F1aBL
# 1kU7dtxSFi8ji
