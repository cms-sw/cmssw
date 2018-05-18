import FWCore.ParameterSet.Config as cms

# MTD rechit producer from uncalibrated rechits
mtdRecHitAlgo = cms.PSet(
    algoName = cms.string("MTDRecHitAlgo"),
    thresholdToKeep = cms.double(1.),
    calibrationConstant = cms.double(0.026041667) # MeV/pC
)
