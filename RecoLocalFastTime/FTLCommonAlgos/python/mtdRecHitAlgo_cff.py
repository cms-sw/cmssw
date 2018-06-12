import FWCore.ParameterSet.Config as cms

# MTD rechit producer from uncalibrated rechits
mtdRecHitAlgo = cms.PSet(
    algoName = cms.string("MTDRecHitAlgo"),
    BTLthresholdToKeep = cms.double(1.),
    BTLcalibrationConstant = cms.double(0.026041667), # MeV/pC
    ETLthresholdToKeep = cms.double(0.5),
    ETLcalibrationConstant = cms.double(1.)
)
