import FWCore.ParameterSet.Config as cms

# FTL "uncalibrated" rechit producer from digis
ftlSimpleUncalibRecHitAlgo = cms.PSet(
    algoName = cms.string("FTLSimpleUncalibRecHitAlgo"),
    adcNbits = cms.uint32(12),
    adcSaturation = cms.double(102),
    toaLSB_ns = cms.double(0.005),
    timeResolutionInNs = cms.double(0.025)
)
