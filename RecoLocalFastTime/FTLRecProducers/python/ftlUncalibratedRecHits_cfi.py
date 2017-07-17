import FWCore.ParameterSet.Config as cms

from RecoLocalFastTime.FTLCommonAlgos.ftlSimpleUncalibRecHitAlgo_cff import ftlSimpleUncalibRecHitAlgo

from SimFastTiming.FastTimingCommon.fastTimeDigitizer_cfi import fastTimeDigitizer

_barrelAlgo = ftlSimpleUncalibRecHitAlgo.clone()
_barrelAlgo.adcNbits = fastTimeDigitizer.barrelDigitizer.ElectronicsSimulation.adcNbits
_barrelAlgo.adcSaturation = fastTimeDigitizer.barrelDigitizer.ElectronicsSimulation.adcSaturation_MIP
_barrelAlgo.toaLSB_ns = fastTimeDigitizer.barrelDigitizer.ElectronicsSimulation.toaLSB_ns
_endcapAlgo = ftlSimpleUncalibRecHitAlgo.clone()
_endcapAlgo.adcNbits = fastTimeDigitizer.endcapDigitizer.ElectronicsSimulation.adcNbits
_endcapAlgo.adcSaturation = fastTimeDigitizer.endcapDigitizer.ElectronicsSimulation.adcSaturation_MIP
_endcapAlgo.toaLSB_ns = fastTimeDigitizer.endcapDigitizer.ElectronicsSimulation.toaLSB_ns

ftlUncalibratedRecHits = cms.EDProducer(
    "FTLUncalibratedRecHitProducer",
    barrel = _barrelAlgo,
    endcap = _endcapAlgo,
    barrelDigis = cms.InputTag('mix:FTLBarrel'),
    endcapDigis = cms.InputTag('mix:FTLEndcap'),
    BarrelHitsName = cms.string('FTLBarrel'),
    EndcapHitsName = cms.string('FTLEndcap')
)
