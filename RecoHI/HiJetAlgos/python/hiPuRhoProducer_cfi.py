import FWCore.ParameterSet.Config as cms

hiPuRhoProducer = cms.EDProducer(
    'HiPuRhoProducer',
    dropZeroTowers = cms.bool(True),
    medianWindowWidth = cms.int32(2), # 0
    minimumTowersFraction = cms.double(0.7),
    nSigmaPU = cms.double(1.0),
    puPtMin = cms.double(15.0),
    rParam = cms.double(.3),
    radiusPU = cms.double(.5),
    src = cms.InputTag('PFTowers'),
    towSigmaCut = cms.double(5.), # -1.
    )
