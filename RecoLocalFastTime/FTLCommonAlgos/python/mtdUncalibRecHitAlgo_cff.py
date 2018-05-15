import FWCore.ParameterSet.Config as cms

# MTD "uncalibrated" rechit producer from digis
mtdUncalibRecHitAlgo = cms.PSet(
    algoName = cms.string("MTDUncalibRecHitAlgo"),
    adcNbits = cms.uint32(12),
    adcSaturation = cms.double(102),
    toaLSB_ns = cms.double(0.005),
    timeResolutionInNs = cms.double(0.025)
)
