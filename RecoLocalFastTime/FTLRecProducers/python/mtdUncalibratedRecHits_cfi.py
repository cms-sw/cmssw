import FWCore.ParameterSet.Config as cms

from SimFastTiming.FastTimingCommon.mtdDigitizer_cfi import mtdDigitizer


_barrelAlgo = cms.PSet(
    algoName = cms.string("BTLUncalibRecHitAlgo"),
    adcNbits = mtdDigitizer.barrelDigitizer.ElectronicsSimulation.adcNbits,
    adcSaturation = mtdDigitizer.barrelDigitizer.ElectronicsSimulation.adcSaturation_MIP,
    toaLSB_ns = mtdDigitizer.barrelDigitizer.ElectronicsSimulation.toaLSB_ns,
    timeResolutionInNs = cms.double(0.025),
    timeCorr_p0 = cms.double( 2.21103),
    timeCorr_p1 = cms.double(-0.933552),
    timeCorr_p2 = cms.double( 0.)
)


_endcapAlgo = cms.PSet(
    algoName      = cms.string("ETLUncalibRecHitAlgo"),
    adcNbits      = mtdDigitizer.endcapDigitizer.ElectronicsSimulation.adcNbits,
    adcSaturation = mtdDigitizer.endcapDigitizer.ElectronicsSimulation.adcSaturation_MIP,
    toaLSB_ns     = mtdDigitizer.endcapDigitizer.ElectronicsSimulation.toaLSB_ns,
    tofDelay      = mtdDigitizer.endcapDigitizer.DeviceSimulation.tofDelay,
    timeResolutionInNs = cms.double(0.025)
)


mtdUncalibratedRecHits = cms.EDProducer(
    "MTDUncalibratedRecHitProducer",
    barrel = _barrelAlgo,
    endcap = _endcapAlgo,
    barrelDigis = cms.InputTag('mix:FTLBarrel'),
    endcapDigis = cms.InputTag('mix:FTLEndcap'),
    BarrelHitsName = cms.string('FTLBarrel'),
    EndcapHitsName = cms.string('FTLEndcap')
)
