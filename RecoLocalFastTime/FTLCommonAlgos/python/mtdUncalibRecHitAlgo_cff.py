import FWCore.ParameterSet.Config as cms

# MTD "uncalibrated" rechit producer from digis
mtdUncalibRecHitAlgo = cms.PSet(
    algoName = cms.string("MTDUncalibRecHitAlgo"),
    adcNbits = cms.uint32(10),
    adcSaturation = cms.double(600),
    toaLSB_ns = cms.double(0.020),
    timeResolutionInNs = cms.double(0.025)
)
