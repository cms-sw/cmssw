import FWCore.ParameterSet.Config as cms

hbhereco = cms.EDFilter(
    "HcalHitReconstructor",
    correctionPhaseNS = cms.double(13.0),
    digiLabel = cms.InputTag("hcalDigis"),
    samplesToAdd = cms.int32(4),
    Subdetector = cms.string('HBHE'),
    firstSample = cms.int32(4),
    correctForPhaseContainment = cms.bool(True),
    correctForTimeslew = cms.bool(True),
    dropZSmarkedPassed = cms.bool(True),

    # Tags for calculating status flags
    correctTiming      = cms.bool(True),
    setNoiseFlags      = cms.bool(True),
    setHSCPFlags       = cms.bool(True),
    setSaturationFlags = cms.bool(True),
    setTimingTrustFlags = cms.bool(False), # timing flags currently only implemented for HF
    
    flagParameters= cms.PSet(nominalPedestal=cms.double(3.0),  #fC
                             hitEnergyMinimum=cms.double(2.0), #GeV
                             hitMultiplicityThreshold=cms.int32(17),
                             pulseShapeParameterSets = cms.VPSet(
    cms.PSet(pulseShapeParameters=cms.vdouble(-100.0, 20.0,-50.0,0.0,-15.0,0.0)),
    cms.PSet(pulseShapeParameters=cms.vdouble( 100.0,2.0e3,-50.0,0.0,-15.0,0.05)),
    cms.PSet(pulseShapeParameters=cms.vdouble( 2.0e3,1.0e6,-50.0,0.0, 85.0,0.0))
    )
                             ),
    saturationParameters=  cms.PSet(maxADCvalue=cms.int32(127)),
    hscpParameters=        cms.PSet(r1Min = cms.double(0.1),
                                    r1Max = cms.double(0.7),
                                    r2Min = cms.double(0.2),
                                    r2Max = cms.double(0.5),
                                    fracLeaderMin = cms.double(0.4),
                                    fracLeaderMax = cms.double(0.7),
                                    slopeMin      = cms.double(-1.5),
                                    slopeMax      = cms.double(-0.6),
                                    outerMin      = cms.double(0.9),
                                    outerMax      = cms.double(1.0),
                                    TimingEnergyThreshold = cms.double(30.))
    )
