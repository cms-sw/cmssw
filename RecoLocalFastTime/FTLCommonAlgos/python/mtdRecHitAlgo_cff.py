import FWCore.ParameterSet.Config as cms

# MTD rechit producer from uncalibrated rechits
mtdRecHitAlgo = cms.PSet(
    algoName = cms.string("MTDRecHitAlgo"),
)
