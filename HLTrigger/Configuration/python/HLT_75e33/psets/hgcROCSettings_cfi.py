import FWCore.ParameterSet.Config as cms

hgcROCSettings = cms.PSet(
    adcNbits = cms.uint32(10),
    adcPulse = cms.vdouble(
        0.0, 0.017, 0.817, 0.163, 0.003,
        0.0
    ),
    adcSaturation_fC = cms.double(100),
    adcThreshold_fC = cms.double(0.672),
    fwVersion = cms.uint32(2),
    jitterConstant_ns = cms.vdouble(0.0004, 0.0004, 0.0004),
    jitterNoise_ns = cms.vdouble(25.0, 25.0, 25.0),
    pulseAvgT = cms.vdouble(
        0.0, 23.42298, 13.16733, 6.41062, 5.03946,
        4.532
    ),
    targetMIPvalue_ADC = cms.uint32(10),
    tdcChargeDrainParameterisation = cms.vdouble(
        -919.13, 365.36, -14.1, 0.2, -21.85,
        49.39, 22.21, 0.8, -0.28, 27.14,
        43.95, 3.89048
    ),
    tdcForToAOnset_fC = cms.vdouble(12.0, 12.0, 12.0),
    tdcNbits = cms.uint32(12),
    tdcOnset_fC = cms.double(60),
    tdcResolutionInPs = cms.double(0.001),
    tdcSaturation_fC = cms.double(10000),
    toaLSB_ns = cms.double(0.0244),
    toaMode = cms.uint32(1)
)