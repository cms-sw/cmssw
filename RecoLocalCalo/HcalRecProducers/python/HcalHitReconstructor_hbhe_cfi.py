import FWCore.ParameterSet.Config as cms

hbheprereco = cms.EDProducer(
    "HcalHitReconstructor",
    correctionPhaseNS = cms.double(13.0),
    digiLabel = cms.InputTag("hcalDigis"),
    Subdetector = cms.string('HBHE'),
    correctForPhaseContainment = cms.bool(True),
    correctForTimeslew = cms.bool(True),
    dropZSmarkedPassed = cms.bool(True),
    firstSample = cms.int32(4),
    samplesToAdd = cms.int32(4),
    tsFromDB = cms.bool(True),

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

    flagParameters= cms.PSet(nominalPedestal=cms.double(3.0),  #fC
                             hitEnergyMinimum=cms.double(1.0), #GeV
                             hitMultiplicityThreshold=cms.int32(17),
                             pulseShapeParameterSets = cms.VPSet(
    cms.PSet(pulseShapeParameters=cms.vdouble(   0.0, 100.0, -50.0, 0.0, -15.0, 0.15)),
    cms.PSet(pulseShapeParameters=cms.vdouble( 100.0, 2.0e3, -50.0, 0.0,  -5.0, 0.05)),
    cms.PSet(pulseShapeParameters=cms.vdouble( 2.0e3, 1.0e6, -50.0, 0.0,  95.0, 0.0 )),
    cms.PSet(pulseShapeParameters=cms.vdouble(-1.0e6, 1.0e6,  45.0, 0.1, 1.0e6, 0.0 )),
    )
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

    pulseShapeParameters = cms.PSet(MinimumChargeThreshold = cms.double(20),
                                    TS4TS5ChargeThreshold = cms.double(70),
                                    TrianglePeakTS = cms.uint32(4),
                                    LinearThreshold = cms.vdouble(20, 70),
                                    LinearCut = cms.vdouble(-2, -0.054),
                                    RMS8MaxThreshold = cms.vdouble(20, 50, 500),
                                    RMS8MaxCut = cms.vdouble(-13, -11.5, -13),
                                    LeftSlopeThreshold = cms.vdouble(23, 30, 40, 95, 140),
                                    LeftSlopeCut = cms.vdouble(2.4, 1.95, 1.7, 1.7, 1.83),
                                    RightSlopeThreshold = cms.vdouble(40, 60, 100, 140, 200),
                                    RightSlopeCut = cms.vdouble(6.2, 5.5, 4.75, 4.38, 4.15),
                                    RightSlopeSmallThreshold = cms.vdouble(60, 80, 110, 140, 200),
                                    RightSlopeSmallCut = cms.vdouble(1.05, 1.135, 1.175, 1.19, 1.17),
                                    MinimumTS4TS5Threshold = cms.double(100),
                                    TS4TS5UpperThreshold = cms.vdouble(70, 90, 100, 400),
                                    TS4TS5UpperCut = cms.vdouble(1, 0.8, 0.75, 0.72),
                                    TS4TS5LowerThreshold = cms.vdouble(100, 120, 160, 200, 300, 500),
                                    TS4TS5LowerCut = cms.vdouble(-1, -0.7, -0.5, -0.4, -0.3, 0.1),
                                    UseDualFit = cms.bool(False),
                                    TriangleIgnoreSlow = cms.bool(False)),

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
                                          )
    )
