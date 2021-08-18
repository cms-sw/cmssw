import FWCore.ParameterSet.Config as cms

from SimFastTiming.FastTimingCommon.mtdDigitizer_cfi import mtdDigitizer


_barrelAlgo = cms.PSet(
    algoName = cms.string("BTLUncalibRecHitAlgo"),
    adcNbits = mtdDigitizer.barrelDigitizer.ElectronicsSimulation.adcNbits,
    adcSaturation = mtdDigitizer.barrelDigitizer.ElectronicsSimulation.adcSaturation_MIP,
    toaLSB_ns = mtdDigitizer.barrelDigitizer.ElectronicsSimulation.toaLSB_ns,
    timeResolutionInNs = cms.string("0.308*pow(x,-0.4175)"), # [ns]
    timeCorr_p0 = cms.double( 2.21103),
    timeCorr_p1 = cms.double(-0.933552),
    timeCorr_p2 = cms.double( 0.),
    c_LYSO = cms.double(13.846235)     # in unit cm/ns
)


_endcapAlgo = cms.PSet(
    algoName      = cms.string("ETLUncalibRecHitAlgo"),
    adcNbits      = mtdDigitizer.endcapDigitizer.ElectronicsSimulation.adcNbits,
    adcSaturation = mtdDigitizer.endcapDigitizer.ElectronicsSimulation.adcSaturation_MIP,
    toaLSB_ns     = mtdDigitizer.endcapDigitizer.ElectronicsSimulation.toaLSB_ns,
    tofDelay      = mtdDigitizer.endcapDigitizer.DeviceSimulation.tofDelay,
    timeResolutionInNs = cms.string("0.039") # [ns]
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

from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(mtdUncalibratedRecHits,
    barrelDigis = 'mixData:FTLBarrel',
    endcapDigis = 'mixData:FTLEndcap',
)
