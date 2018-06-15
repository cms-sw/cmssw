import FWCore.ParameterSet.Config as cms

from RecoLocalFastTime.FTLCommonAlgos.mtdRecHitAlgo_cff import mtdRecHitAlgo

_barrelAlgo = mtdRecHitAlgo.clone()
_endcapAlgo = mtdRecHitAlgo.clone()

mtdRecHits = cms.EDProducer(
    "MTDRecHitProducer",
    barrel = _barrelAlgo,
    endcap = _endcapAlgo,
    barrelUncalibratedRecHits = cms.InputTag('mtdUncalibratedRecHits:FTLBarrel'),
    endcapUncalibratedRecHits = cms.InputTag('mtdUncalibratedRecHits:FTLEndcap'),
    BarrelHitsName = cms.string('FTLBarrel'),
    EndcapHitsName = cms.string('FTLEndcap'),
    BTLthresholdToKeep = cms.double(1.),
    BTLcalibrationConstant = cms.double(0.026041667), # MeV/pC
    ETLthresholdToKeep = cms.double(0.5),
    ETLcalibrationConstant = cms.double(1.)
)
