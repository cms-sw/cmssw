import FWCore.ParameterSet.Config as cms

# MTD "uncalibrated" rechit producer from digis
mtdUncalibRecHitAlgo = cms.PSet(
    algoName = cms.string("MTDUncalibRecHitAlgo"),
    BTLadcNbits = cms.uint32(10),
    BTLadcSaturation = cms.double(600),
    BTLtoaLSB_ns = cms.double(0.020),
    BTLtimeResolutionInNs = cms.double(0.025),
    ETLadcNbits = cms.uint32(12),
    ETLadcSaturation = cms.double(102),
    ETLtoaLSB_ns = cms.double(0.005),
    ETLtimeResolutionInNs = cms.double(0.025)
)
