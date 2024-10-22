import FWCore.ParameterSet.Config as cms

hgchebackDigitizer = cms.PSet(
    NoiseGeneration_Method = cms.bool(True),
    accumulatorType = cms.string('HGCDigiProducer'),
    bxTime = cms.double(25),
    digiCfg = cms.PSet(
        algo = cms.uint32(2),
        doTimeSamples = cms.bool(False),
        feCfg = cms.PSet(
            adcNbits = cms.uint32(10),
            adcPulse = cms.vdouble(
                0.0, 0.017, 0.817, 0.163, 0.003,
                0.0
            ),
            adcSaturation_fC = cms.double(68.75),
            adcThreshold_fC = cms.double(0.5),
            fwVersion = cms.uint32(2),
            jitterConstant_ns = cms.vdouble(0.0004, 0.0004, 0.0004),
            jitterNoise_ns = cms.vdouble(25.0, 25.0, 25.0),
            pulseAvgT = cms.vdouble(
                0.0, 23.42298, 13.16733, 6.41062, 5.03946,
                4.532
            ),
            targetMIPvalue_ADC = cms.uint32(15),
            tdcChargeDrainParameterisation = cms.vdouble(
                -919.13, 365.36, -14.1, 0.2, -21.85,
                49.39, 22.21, 0.8, -0.28, 27.14,
                43.95, 3.89048
            ),
            tdcForToAOnset_fC = cms.vdouble(12.0, 12.0, 12.0),
            tdcNbits = cms.uint32(12),
            tdcOnset_fC = cms.double(55),
            tdcResolutionInPs = cms.double(0.001),
            tdcSaturation_fC = cms.double(1000),
            toaLSB_ns = cms.double(0.0244),
            toaMode = cms.uint32(1)
        ),
        keV2MIP = cms.double(0.00148148148148),
        nPEperMIP = cms.double(21.0),
        nTotalPE = cms.double(7500),
        noise = cms.PSet(
            refToPSet_ = cms.string('HGCAL_noise_heback')
        ),
        scaleBySipmArea = cms.bool(False),
        scaleByTileArea = cms.bool(False),
        sdPixels = cms.double(1e-06),
        sipmMap = cms.string('SimCalorimetry/HGCalSimProducers/data/sipmParams_geom-10.txt'),
        thresholdFollowsMIP = cms.bool(True),
        xTalk = cms.double(0.01)
    ),
    digiCollection = cms.string('HGCDigisHEback'),
    digitizationType = cms.uint32(1),
    geometryType = cms.uint32(1),
    hitCollection = cms.string('HGCHitsHEback'),
    makeDigiSimLinks = cms.bool(False),
    maxSimHitsAccTime = cms.uint32(100),
    premixStage1 = cms.bool(False),
    premixStage1MaxCharge = cms.double(1000000.0),
    premixStage1MinCharge = cms.double(0),
    tofDelay = cms.double(1),
    useAllChannels = cms.bool(True),
    verbosity = cms.untracked.uint32(0)
)