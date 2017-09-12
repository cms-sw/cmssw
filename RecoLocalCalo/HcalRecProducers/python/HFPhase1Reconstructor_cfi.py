import FWCore.ParameterSet.Config as cms

hfreco = cms.EDProducer("HFPhase1Reconstructor",
    # Label for the input HFPreRecHitCollection
    inputLabel = cms.InputTag("hfprereco"),

    # Change the following to True in order to use the channel
    # status from the DB
    useChannelQualityFromDB = cms.bool(True),

    # Change the following to True when the status becomes
    # available in the DB for both anodes. If this parameter
    # is set to False then it is assumed that the status of
    # both anodes is given by the channel at depth 1 and 2.
    checkChannelQualityForDepth3and4 = cms.bool(True),

    # Configure the reconstruction algorithm
    algorithm = cms.PSet(
        Class = cms.string("HFFlexibleTimeCheck"),

        # Timing cuts: pass everything for now
        tlimits = cms.vdouble(-1000.0, 1000.0,
                              -1000.0, 1000.0),

        # Linear mapping of the array with dimensions [13][2].
        # The first dimension is 2*HFAnodeStatus::N_POSSIBLE_STATES - 1.
        energyWeights = cms.vdouble(
            1.0, 1.0,  # {OK, OK} anode status
            1.0, 0.0,  # {OK, NOT_DUAL}
            1.0, 0.0,  # {OK, NOT_READ_OUT}
            2.0, 0.0,  # {OK, HARDWARE_ERROR}
            2.0, 0.0,  # {OK, FLAGGED_BAD}
            2.0, 0.0,  # {OK, FAILED_TIMING}
            1.0, 0.0,  # {OK, FAILED_OTHER}
            0.0, 1.0,  # {NOT_DUAL, OK}
            0.0, 1.0,  # {NOT_READ_OUT, OK}
            0.0, 2.0,  # {HARDWARE_ERROR, OK}
            0.0, 2.0,  # {FLAGGED_BAD, OK}
            0.0, 2.0,  # {FAILED_TIMING, OK}
            0.0, 1.0   # {FAILED_OTHER, OK}
        ),

        # Into which byte (0, 1, or 2) of the aux word the sample
        # of interest ADC will be placed?
        soiPhase = cms.uint32(1),

        # Time shift added to all "normal" QIE10 TDC time measurements
        timeShift = cms.double(0.0),

        # Rise and fall time of the rechit will be set to these values
        # if neither anode has valid TDC info
        triseIfNoTDC = cms.double(-100.0),
        tfallIfNoTDC = cms.double(-101.0),

        # Charge limits for special TDC values. If the anode charge is
        # below such a limit, the anode will participate in the energy
        # reconstruction even if its TDC undershoots/overshoots. These
        # global limits are in addition to those per channel limits in
        # the database (effectively, the larger limit is used).
        minChargeForUndershoot = cms.double(1.0e10),
        minChargeForOvershoot = cms.double(1.0e10),

        # Do not construct rechits with problems
        rejectAllFailures = cms.bool(True)
    ),

    # Reconstruction algorithm data to fetch from DB, if any
    algoConfigClass = cms.string("HFPhase1PMTParams"),

    # Turn on/off the noise cleanup algorithms
    setNoiseFlags = cms.bool(True),

    # Parameters for the S9S1 test.
    #
    #   optimumSlopes are slopes for each of the |ieta| values
    #   29, 30, .... ,41  (although |ieta|=29 is not used in
    #   current S9S1 formulation)
    #
    #   energy and ET params are thresholds for each |ieta|
    #
    S9S1stat = cms.PSet(
        # WARNING!  ONLY LONG PARAMETERS ARE USED IN DEFAULT RECO; SHORT S9S1 IS NOT USED!
        short_optimumSlope   = cms.vdouble([-99999,0.0164905,0.0238698,0.0321383,
                                            0.041296,0.0513428,0.0622789,0.0741041,
                                            0.0868186,0.100422,0.135313,0.136289,
                                            0.0589927]),

        # Short energy cut is 129.9 - 6.61*|ieta|+0.1153*|ieta|^2
        shortEnergyParams    = cms.vdouble([35.1773, 35.37, 35.7933, 36.4472,
                                            37.3317, 38.4468, 39.7925, 41.3688,
                                            43.1757, 45.2132, 47.4813, 49.98,
                                            52.7093]),
        shortETParams        = cms.vdouble([0,0,0,0,
                                            0,0,0,0,
                                            0,0,0,0,0]),

        long_optimumSlope    = cms.vdouble([-99999,0.0164905,0.0238698,0.0321383,
                                            0.041296,0.0513428,0.0622789,0.0741041,
                                            0.0868186,0.100422,0.135313,0.136289,
                                            0.0589927]),

        # Long energy cut is 162.4-10.9*abs(ieta)+0.21*ieta*ieta
        longEnergyParams     = cms.vdouble([43.5, 45.7, 48.32, 51.36,
                                            54.82, 58.7, 63.0, 67.72,
                                            72.86, 78.42, 84.4, 90.8,
                                            97.62]),
        longETParams         = cms.vdouble([0,0,0,0,
                                            0,0,0,0,
                                            0,0,0,0,0]),

        HcalAcceptSeverityLevel = cms.int32(9), # allow hits with severity up to AND INCLUDING 9
        isS8S1                  = cms.bool(False),
    ),

    # Parameters for the S8S1 test. Sets the HFS8S1Ratio Bit (bit 3).
    #
    #   energy and ET params are coefficients for
    #   energy/ET thresholds, parameterized in ieta
    #
    S8S1stat = cms.PSet(
        short_optimumSlope   = cms.vdouble([0.30, # ieta=29 is a special case
                                            0.10, 0.10, 0.10, 0.10,
                                            0.10, 0.10, 0.10, 0.10,
                                            0.10, 0.10, 0.10, 0.10]),

        # Short energy cut is 40 for ieta=29, 100 otherwise
        shortEnergyParams    = cms.vdouble([40,
                                            100,100,100,100,
                                            100,100,100,100,
                                            100,100,100,100]),
        shortETParams        = cms.vdouble([0,0,0,0,
                                            0,0,0,0,
                                            0,0,0,0,0]),

        long_optimumSlope   = cms.vdouble([0.30, # ieta=29 is a special case
                                           0.10, 0.10, 0.10, 0.10,
                                           0.10, 0.10, 0.10, 0.10,
                                           0.10, 0.10, 0.10, 0.10]),

        # Long energy cut is 40 for ieta=29, 100 otherwise
        longEnergyParams    = cms.vdouble([40,
                                           100,100,100,100,
                                           100,100,100,100,
                                           100,100,100,100]),
        longETParams        = cms.vdouble([0,0,0,0,
                                           0,0,0,0,
                                           0,0,0,0,0]),

        HcalAcceptSeverityLevel  = cms.int32(9), # allow hits with severity up to AND INCLUDING 9
        isS8S1                   = cms.bool(True),
    ),

    # Parameters for the Parameterized Energy Threshold (PET) test.
    #
    #   short_R, long_R are coefficients of R threshold,
    #   parameterized in *ENERGY*:  R_thresh = [0]+[1]*energy+[2]*energy^2+...
    #
    #   As of March 2010, the R threshold is a simple fixed value:
    #   R>0.98, with separate params for |ieta|=29
    #
    #   Energy and ET params are energy and ET cuts for each |ieta| 29 -> 41
    #
    PETstat = cms.PSet(
        short_R = cms.vdouble([0.8]),  # new default ratio cut:  R>0.8

        # Short energy cut is 129.9 - 6.61*|ieta|+0.1153*|ieta|^2
        shortEnergyParams    = cms.vdouble([35.1773, 35.37, 35.7933, 36.4472,
                                            37.3317, 38.4468, 39.7925, 41.3688,
                                            43.1757, 45.2132, 47.4813, 49.98,
                                            52.7093]),
        shortETParams        = cms.vdouble([0,0,0,0,
                                            0,0,0,0,
                                            0,0,0,0,0]),

        long_R  = cms.vdouble([0.98]),  # default ratio cut:  R>0.98

        # Long energy cut is 162.4-10.9*abs(ieta)+0.21*ieta*ieta
        longEnergyParams    = cms.vdouble([43.5, 45.7, 48.32, 51.36,
                                           54.82, 58.7, 63.0, 67.72,
                                           72.86, 78.42, 84.4, 90.8,
                                           97.62]),
        longETParams        = cms.vdouble([0,0,0,0,
                                           0,0,0,0,
                                           0,0,0,0,0]),

        short_R_29 = cms.vdouble([0.8]),
        long_R_29  = cms.vdouble([0.8]), # should move from 0.98 to 0.8?
        HcalAcceptSeverityLevel = cms.int32(9), # allow hits with severity up to AND INCLUDING 9
    )
)
