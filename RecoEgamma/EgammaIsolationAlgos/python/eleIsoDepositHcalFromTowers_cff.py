import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaIsolationAlgos.eleHcalExtractorBlocks_cff import *

eleIsoDepositHcalFromTowers = cms.EDProducer("CandIsoDepositProducer",
    src = cms.InputTag("gedGsfElectrons"),
    trackType = cms.string('candidate'),
    MultipleDepositsFlag = cms.bool(False),
    ExtractorPSet = cms.PSet(EleIsoHcalFromTowersExtractorBlock)
)

#next two seperate out the hcal depths in the endcap
#note that by defination hcal depth 1 + hcal depth 2 = hcal
eleIsoDepositHcalDepth1FromTowers= cms.EDProducer("CandIsoDepositProducer",
    src = cms.InputTag("gedGsfElectrons"),
    MultipleDepositsFlag = cms.bool(False),
    trackType = cms.string('candidate'),
    ExtractorPSet = cms.PSet( EleIsoHcalFromTowersExtractorBlock )
)
eleIsoDepositHcalDepth1FromTowers.ExtractorPSet.hcalDepth = cms.int32(1)

eleIsoDepositHcalDepth2FromTowers = cms.EDProducer("CandIsoDepositProducer",
    src = cms.InputTag("gedGsfElectrons"),
    MultipleDepositsFlag = cms.bool(False),
    trackType = cms.string('candidate'),
    ExtractorPSet = cms.PSet( EleIsoHcalFromTowersExtractorBlock )
)
eleIsoDepositHcalDepth2FromTowers.ExtractorPSet.hcalDepth = cms.int32(2)
