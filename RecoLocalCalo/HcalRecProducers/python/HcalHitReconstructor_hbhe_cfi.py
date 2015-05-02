import FWCore.ParameterSet.Config as cms

hbheprereco = cms.EDProducer(
    "HcalHitReconstructor",
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

    # Disable negative energy filter pending db support
    setNegativeFlags          = cms.bool(False),

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
                                    LinearThreshold = cms.vdouble(20, 100, 100000),
                                    LinearCut = cms.vdouble(-3, -0.054, -0.054),
                                    RMS8MaxThreshold = cms.vdouble(20, 100, 100000),
                                    RMS8MaxCut = cms.vdouble(-13.5, -11.5, -11.5),
                                    LeftSlopeThreshold = cms.vdouble(250, 500, 100000),
                                    LeftSlopeCut = cms.vdouble(5, 2.55, 2.55),
                                    RightSlopeThreshold = cms.vdouble(250, 400, 100000),
                                    RightSlopeCut = cms.vdouble(5, 4.15, 4.15),
                                    RightSlopeSmallThreshold = cms.vdouble(150, 200, 100000),
                                    RightSlopeSmallCut = cms.vdouble(1.08, 1.16, 1.16),
                                    MinimumTS4TS5Threshold = cms.double(100),
                                    TS4TS5UpperThreshold = cms.vdouble(70, 90, 100, 400),
                                    TS4TS5UpperCut = cms.vdouble(1, 0.8, 0.75, 0.72),
                                    TS4TS5LowerThreshold = cms.vdouble(100, 120, 160, 200, 300, 500),
                                    TS4TS5LowerCut = cms.vdouble(-1, -0.7, -0.5, -0.4, -0.3, 0.1),
                                    UseDualFit = cms.bool(True),
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
                                          ),
    applyPedConstraint    = cms.bool(True),
    applyTimeConstraint   = cms.bool(True),
    applyPulseJitter      = cms.bool(False),  
    applyUnconstrainedFit = cms.bool(False),   #Turn on original Method 2
    applyTimeSlew         = cms.bool(True),   #units
    ts4Min                = cms.double(5.),   #fC
    ts4Max                = cms.double(500.),   #fC
    pulseJitter           = cms.double(1.),   #GeV/bin
    meanTime              = cms.double(-5.5), #ns
    timeSigma             = cms.double(5.),  #ns
    meanPed               = cms.double(0.),   #GeV
    pedSigma              = cms.double(0.5),  #GeV
    noise                 = cms.double(1),    #fC
    timeMin               = cms.double(-18),  #ns
    timeMax               = cms.double( 7),  #ns
    ts3chi2               = cms.double(5.),   #chi2 (not used)
    ts4chi2               = cms.double(15.),  #chi2 for triple pulse 
    ts345chi2             = cms.double(100.), #chi2 (not used)
    chargeMax             = cms.double(6.),    #Charge cut (fC) for uncstrianed Fit 
    fitTimes              = cms.int32(1)       # -1 means no constraint on number of fits per channel
    )
