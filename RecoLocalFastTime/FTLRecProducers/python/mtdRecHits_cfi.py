import FWCore.ParameterSet.Config as cms

_barrelAlgo = cms.PSet(
    algoName = cms.string("MTDRecHitAlgo"),
    thresholdToKeep = cms.vdouble(1., 1.),          # MeV
    calibrationConstant = cms.vdouble(0.03125, 0.03125), # MeV/pC
)


_endcapAlgo = cms.PSet(
    algoName = cms.string("MTDRecHitAlgo"),
    thresholdToKeep = cms.vdouble(0.0425, 0.005),    # MeV
    calibrationConstant = cms.vdouble(0.085, 0.01), # MeV/MIP
)


mtdRecHits = cms.EDProducer(
    "MTDRecHitProducer",
    barrel = _barrelAlgo,
    endcap = _endcapAlgo,
    barrelUncalibratedRecHits = cms.InputTag('mtdUncalibratedRecHits:FTLBarrel'),
    endcapUncalibratedRecHits = cms.InputTag('mtdUncalibratedRecHits:FTLEndcap'),
    BarrelHitsName = cms.string('FTLBarrel'),
    EndcapHitsName = cms.string('FTLEndcap'),
)
