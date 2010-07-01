import FWCore.ParameterSet.Config as cms

hbhereco = cms.EDProducer(
    "HcalHitReconstructor",
    correctionPhaseNS = cms.double(13.0),
    digiLabel = cms.InputTag("hcalDigis"),
    samplesToAdd = cms.int32(4),
    Subdetector = cms.string('HBHE'),
    firstSample = cms.int32(4),
    correctForPhaseContainment = cms.bool(True),
    correctForTimeslew = cms.bool(True),
    dropZSmarkedPassed = cms.bool(True),

    # Set offset between firstSample value and
    # first sample to be stored in aux word
    firstAuxOffset = cms.int32(0),
    
    # Tags for calculating status flags
    correctTiming             = cms.bool(True),
    setNoiseFlags             = cms.bool(True),
    setHSCPFlags              = cms.bool(True),
    setSaturationFlags        = cms.bool(True),
    setTimingShapedCutsFlags  = cms.bool(True),
    setTimingTrustFlags       = cms.bool(False), # timing flags currently only implemented for HF
    
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
    # shaped cut parameters are pairs of (energy, time threshold) values
    # These must be ordered by increaseing energy!
    timingshapedcutsParameters = cms.PSet(tfilterEnvelope=cms.vdouble(4.00,12.04,
                                                                      13.00,10.56,
                                                                      23.50,8.82,
                                                                      37.00,7.38,
                                                                      56.00,6.30,
                                                                      81.00,5.64,
                                                                      114.50,5.44,
                                                                      175.50,5.38,
                                                                      350.50,5.14),
                                          ignorelowest  = cms.bool(True), # ignores hits with energies below lowest envelope threshold
                                          ignorehighest = cms.bool(False), # ignores hits with energies above highest envelope threshold
                                          win_offset    = cms.double(0.),
                                          win_gain      = cms.double(1.))

    )
