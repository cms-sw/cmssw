import FWCore.ParameterSet.Config as cms

from RecoLocalFastTime.FTLCommonAlgos.mtdUncalibRecHitAlgo_cff import mtdUncalibRecHitAlgo

from SimFastTiming.FastTimingCommon.mtdDigitizer_cfi import mtdDigitizer

_barrelAlgo = mtdUncalibRecHitAlgo.clone()
_barrelAlgo.adcNbits = mtdDigitizer.barrelDigitizer.ElectronicsSimulation.adcNbits
_barrelAlgo.adcSaturation = mtdDigitizer.barrelDigitizer.ElectronicsSimulation.adcSaturation_MIP
_barrelAlgo.toaLSB_ns = mtdDigitizer.barrelDigitizer.ElectronicsSimulation.toaLSB_ns
_endcapAlgo = mtdUncalibRecHitAlgo.clone()
_endcapAlgo.adcNbits = mtdDigitizer.endcapDigitizer.ElectronicsSimulation.adcNbits
_endcapAlgo.adcSaturation = mtdDigitizer.endcapDigitizer.ElectronicsSimulation.adcSaturation_MIP
_endcapAlgo.toaLSB_ns = mtdDigitizer.endcapDigitizer.ElectronicsSimulation.toaLSB_ns

mtdUncalibratedRecHits = cms.EDProducer(
    "MTDUncalibratedRecHitProducer",
    barrel = _barrelAlgo,
    endcap = _endcapAlgo,
    barrelDigis = cms.InputTag('mix:FTLBarrel'),
    endcapDigis = cms.InputTag('mix:FTLEndcap'),
    BarrelHitsName = cms.string('FTLBarrel'),
    EndcapHitsName = cms.string('FTLEndcap')
)
