import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaIsolationAlgos.gamHcalExtractorBlocks_cff import *

gamIsoDepositHcalFromTowers = cms.EDProducer("CandIsoDepositProducer",
    src = cms.InputTag("photons"),
    trackType = cms.string('candidate'),
    MultipleDepositsFlag = cms.bool(False),
    ExtractorPSet = cms.PSet(GamIsoHcalFromTowersExtractorBlock)
)

#next two seperate out the hcal depths in the endcap
#note that by defination hcal depth 1 + hcal depth 2 = hcal
gamIsoDepositHcalDepth1FromTowers= cms.EDProducer("CandIsoDepositProducer",
    src = cms.InputTag("photons"),
    MultipleDepositsFlag = cms.bool(False),
    trackType = cms.string('candidate'),
    ExtractorPSet = cms.PSet( GamIsoHcalFromTowersExtractorBlock )
)
gamIsoDepositHcalDepth1FromTowers.ExtractorPSet.hcalDepth = cms.int32(1)

gamIsoDepositHcalDepth2FromTowers = cms.EDProducer("CandIsoDepositProducer",
    src = cms.InputTag("photons"),
    MultipleDepositsFlag = cms.bool(False),
    trackType = cms.string('candidate'),
    ExtractorPSet = cms.PSet( GamIsoHcalFromTowersExtractorBlock )
)
gamIsoDepositHcalDepth2FromTowers.ExtractorPSet.hcalDepth = cms.int32(2)

