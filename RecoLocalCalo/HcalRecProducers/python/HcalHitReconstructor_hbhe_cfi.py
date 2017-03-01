import FWCore.ParameterSet.Config as cms
import RecoLocalCalo.HcalRecProducers.HBHEMethod3Parameters_cfi as method3
import RecoLocalCalo.HcalRecProducers.HBHEMethod2Parameters_cfi as method2
import RecoLocalCalo.HcalRecProducers.HBHEPulseShapeFlagSetter_cfi as pulseShapeFlag
import RecoLocalCalo.HcalRecProducers.HBHEStatusBitSetter_cfi as hbheStatusFlag

hbheprereco = cms.EDProducer(
    "HcalHitReconstructor",
    method3.m3Parameters,
    method2.m2Parameters,
    correctionPhaseNS = cms.double(6.0),
    digiLabel = cms.InputTag("hcalDigis"),
    Subdetector = cms.string('HBHE'),
    correctForPhaseContainment = cms.bool(True),
    correctForTimeslew = cms.bool(True),
    dropZSmarkedPassed = cms.bool(True),
    firstSample = cms.int32(4),
    samplesToAdd = cms.int32(2),
    tsFromDB = cms.bool(True),
    recoParamsFromDB = cms.bool(True),
    useLeakCorrection = cms.bool(False),
    dataOOTCorrectionName = cms.string("HBHE"),
    dataOOTCorrectionCategory = cms.string("Data"),
    mcOOTCorrectionName = cms.string("HBHE"),
    mcOOTCorrectionCategory = cms.string("MC"),
    puCorrMethod = cms.int32(2),

    # Set time slice for first digi to be stored in aux word
    # (HBHE uses time slices 4-7 for reco)
    firstAuxTS = cms.int32(4),

    # Tags for calculating status flags
    correctTiming             = cms.bool(True),
    setNoiseFlags             = cms.bool(True),
    setHSCPFlags              = cms.bool(True),
    setSaturationFlags        = cms.bool(True),
    setTimingShapedCutsFlags  = cms.bool(True),
    setTimingTrustFlags       = cms.bool(False), # timing flags currently only implemented for HF
    setPulseShapeFlags        = cms.bool(True),

    # Enable negative energy filter
    setNegativeFlags          = cms.bool(True),

    flagParameters = cms.PSet(
        hbheStatusFlag.qie8Config
    ),

    saturationParameters=  cms.PSet(maxADCvalue=cms.int32(127)),
    hscpParameters=        cms.PSet(r1Min = cms.double(0.15),  # was 0.1
                                    r1Max = cms.double(1.0),   # was 0.7
                                    r2Min = cms.double(0.1),   # was 0.1
                                    r2Max = cms.double(0.5),
                                    fracLeaderMin = cms.double(0.4),
                                    fracLeaderMax = cms.double(0.7),
                                    slopeMin      = cms.double(-1.5),
                                    slopeMax      = cms.double(-0.6),
                                    outerMin      = cms.double(0.), # was 0.
                                    outerMax      = cms.double(0.1), # was 0.1
                                    TimingEnergyThreshold = cms.double(30.)),

    pulseShapeParameters = cms.PSet(
        pulseShapeFlag.qie8Parameters
    ),

    # shaped cut parameters are triples of (energy, low time threshold, high time threshold) values.
    # The low and high thresholds must straddle zero (i.e., low<0, high>0); use win_offset to shift.
    # win_gain is applied to both threshold values before win_offset.
    # Energy ordering is no longer required on input, but guaranteed by the software.
    #  note that energies are rounded to the nearest GeV.
    #
    timingshapedcutsParameters = cms.PSet(tfilterEnvelope=cms.vdouble(  50.0,  -2.0,  4.25,
                                                                        52.0,  -2.0,  4.09,
                                                                        54.0,  -2.0,  3.95,
                                                                        56.0,  -2.0,  3.82,
                                                                        58.0,  -2.0,  3.71,
                                                                        60.0,  -2.0,  3.60,
                                                                        63.0,  -2.0,  3.46,
                                                                        66.0,  -2.0,  3.33,
                                                                        69.0,  -2.0,  3.22,
                                                                        73.0,  -2.0,  3.10,
                                                                        77.0,  -2.0,  2.99,
                                                                        82.0,  -2.0,  2.87,
                                                                        88.0,  -2.0,  2.75,
                                                                        95.0,  -2.0,  2.64,
                                                                        103.0, -2.0,  2.54,
                                                                        113.0, -2.0,  2.44,
                                                                        127.0, -2.0,  2.33,
                                                                        146.0, -2.0,  2.23,
                                                                        176.0, -2.0,  2.13,
                                                                        250.0, -2.0,  2.00 ),
                                          win_offset = cms.double(0.0),
                                          win_gain   = cms.double(3.0),
                                          ignorelowest=cms.bool(True),
                                          ignorehighest=cms.bool(False)
                                          ),
)
