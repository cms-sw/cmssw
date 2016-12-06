import FWCore.ParameterSet.Config as cms

# FTL rechit producer from uncalibrated rechits
ftlSimpleRecHitAlgo = cms.EDProducer(
    algoName = cms.string("FTLSimpleRecHitAlgo"),
    thresholdToKeep = cms.double(0.5),
    calibrationConstant = cms.double(1.0)
)
