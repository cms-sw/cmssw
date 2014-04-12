import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaIsolationAlgos.eleHcalExtractorBlocks_cff import *

#This module is defined for the user who would like to study HCAL
#isolation with RecHits.
#Currently, EGamma POG is recommending HCAL Isolation with Towers
eleIsoDepositHcalFromHits = cms.EDProducer("CandIsoDepositProducer",
    src = cms.InputTag("gedGsfElectrons"),
    trackType = cms.string('candidate'),
    MultipleDepositsFlag = cms.bool(False),
    ExtractorPSet = cms.PSet(EleIsoHcalFromHitsExtractorBlock)
)



