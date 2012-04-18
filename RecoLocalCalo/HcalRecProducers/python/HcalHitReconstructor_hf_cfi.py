import FWCore.ParameterSet.Config as cms
import string # use for setting flag masks based on boolean bits

hfreco = cms.EDProducer("HcalHitReconstructor",
                        correctionPhaseNS = cms.double(0.0),
                        digiLabel = cms.InputTag("hcalDigis"),
                        Subdetector = cms.string('HF'),
                        correctForPhaseContainment = cms.bool(False),
                        correctForTimeslew = cms.bool(False),
                        dropZSmarkedPassed = cms.bool(True),
   		        firstSample = cms.int32(2),
                        samplesToAdd = cms.int32(2),	
                        tsFromDB = cms.bool(True),

                        correctTiming = cms.bool(True),
                        # Set time slice for first digi to be stored in aux word 
                        firstAuxTS = cms.int32(1),

                        # Tags for calculating status flags
                        setNoiseFlags = cms.bool(True),
                        setHSCPFlags  = cms.bool(True),
                        setSaturationFlags = cms.bool(True),
                        setTimingTrustFlags = cms.bool(True),
                        setPulseShapeFlags = cms.bool(False),  # not yet defined for HF

                        digistat= cms.PSet(HFdigiflagFirstSample     = cms.int32(1),  # These may be different from samples used for reconstruction
                                           HFdigiflagSamplesToAdd    = cms.int32(3),  # Use 3 TS for 75-ns running
                                           HFdigiflagExpectedPeak    = cms.int32(2), # expected TS position of pulse peak
                                           HFdigiflagMinEthreshold  = cms.double(40), # minimum energy required to be flagged as noisy
                                           # Following parameters are used for determining
                                           # minimum threshold fC(peak)/sum_fC(HFsamplesToAdd) > [0] - exp([1]+[2]*Energy)
                                           HFdigiflagCoef0           = cms.double(0.93),
                                           HFdigiflagCoef1           = cms.double(-0.38275),
                                           HFdigiflagCoef2           = cms.double(-0.012667)
                                           ),

                        # Window Parameters require that reconstructed time occurs min and max window time
                        # Time Parameters are expressed as coefficients in polynomial expansion in 1/energy:  [0]+[1]/E + ...
                        HFInWindowStat = cms.PSet(hflongMinWindowTime=cms.vdouble([-10]),
                                                  hflongMaxWindowTime=cms.vdouble([10]),
                                                  hflongEthresh=cms.double(40.),
                                                  hfshortMinWindowTime=cms.vdouble([-12]),
                                                  hfshortMaxWindowTime=cms.vdouble([10]),
                                                  hfshortEthresh=cms.double(40.),
                                                  ),


                        # Parameters for Using S9S1 Test
                        #     optimumSlopes are slopes for each of the |ieta| values 29, 30, .... ,41  (although |ieta|=29 is not used in current S9S1 formulation)

                        #     energy and ET params are thresholds for each |ieta|
                        S9S1stat = cms.PSet(
    # WARNING!  ONLY LONG PARAMETERS ARE USED IN DEFAULT RECO; SHORT S9S1 IS NOT USED!
    short_optimumSlope       = cms.vdouble([-99999,0.0164905,0.0238698,0.0321383,
                                            0.041296,0.0513428,0.0622789,0.0741041,
                                            0.0868186,0.100422,0.135313,0.136289,
                                            0.0589927]),

    # Short energy cut is 129.9 - 6.61*|ieta|+0.1153*|ieta|^2
    shortEnergyParams        = cms.vdouble([35.1773, 35.37, 35.7933, 36.4472,
                                            37.3317, 38.4468, 39.7925, 41.3688,
                                            43.1757, 45.2132, 47.4813, 49.98,
                                            52.7093]),
    shortETParams            = cms.vdouble([0,0,0,0,
                                            0,0,0,0,
                                            0,0,0,0,0]),

    long_optimumSlope       = cms.vdouble([-99999,0.0164905,0.0238698,0.0321383,
                                           0.041296,0.0513428,0.0622789,0.0741041,
                                           0.0868186,0.100422,0.135313,0.136289,
                                           0.0589927]),

    # Long energy cut is 162.4-10.9*abs(ieta)+0.21*ieta*ieta
    longEnergyParams        = cms.vdouble([43.5, 45.7, 48.32, 51.36,
                                           54.82, 58.7, 63.0, 67.72,
                                           72.86, 78.42, 84.4, 90.8,
                                           97.62]),
    longETParams            = cms.vdouble([0,0,0,0,
                                           0,0,0,0,
                                           0,0,0,0,0]),

    flagsToSkip              = cms.int32(string.atoi('11000',2)), # HFPET (bit 4), HFDigiTime (bit 1) and HFS8S1Ratio (bit 3) should be skipped, but not HFDigiTime in MC
    isS8S1                   = cms.bool(False),
    ),


                        # Parameters for Using S8S1 Test
                        # Sets the HFS8S1Ratio Bit (bit 3)

                        #     energy and ET params are coefficients for energy/ET thresholds, parameterized in ieta
                        S8S1stat = cms.PSet(
    short_optimumSlope       = cms.vdouble([0.30, # ieta=29 is a special case
                                            0.10, 0.10, 0.10, 0.10,
                                            0.10, 0.10, 0.10, 0.10,
                                            0.10, 0.10, 0.10, 0.10]),

    # Short energy cut is 40 for ieta=29, 100 otherwise
    shortEnergyParams        = cms.vdouble([40,
                                            100,100,100,100,
                                            100,100,100,100,
                                            100,100,100,100]),
    shortETParams            = cms.vdouble([0,0,0,0,
                                            0,0,0,0,
                                            0,0,0,0,0]),

    long_optimumSlope       = cms.vdouble([0.30, # ieta=29 is a special case
                                           0.10, 0.10, 0.10, 0.10,
                                           0.10, 0.10, 0.10, 0.10,
                                           0.10, 0.10, 0.10, 0.10]),
    # Long energy cut is 40 for ieta=29, 100 otherwise
    longEnergyParams        = cms.vdouble([40,
                                           100,100,100,100,
                                           100,100,100,100,
                                           100,100,100,100]),
    longETParams            = cms.vdouble([0,0,0,0,
                                           0,0,0,0,
                                           0,0,0,0,0]),

    flagsToSkip              = cms.int32(string.atoi('10000',2)), # HFPET (bit 4) and HFDigiTime (bit 1) should be skipped, but not HFDigiTime in MC
    isS8S1                   = cms.bool(True),
    ),


                        # Parameters for Using Parameterized Energy Threshold (PET) test
                        #  short_R, long_R are coefficients of R threshold, parameterized in *ENERGY*:  R_thresh = [0]+[1]*energy+[2]*energy^2+...
                        #  As of March 2010, the R threshold is a simple fixed value:  R>0.98, with separate params for |ieta|=29
                        #  Energy and ET params are energy and ET cuts for each |ieta| 29 -> 41

                        PETstat = cms.PSet(

    short_R = cms.vdouble([0.8]),  # new default ratio cut:  R>0.8
    # Short energy cut is 129.9 - 6.61*|ieta|+0.1153*|ieta|^2
    shortEnergyParams        = cms.vdouble([35.1773, 35.37, 35.7933, 36.4472,
                                            37.3317, 38.4468, 39.7925, 41.3688,
                                            43.1757, 45.2132, 47.4813, 49.98,
                                            52.7093]),
    shortETParams            = cms.vdouble([0,0,0,0,
                                            0,0,0,0,
                                            0,0,0,0,0]),

    long_R  = cms.vdouble([0.98]),  # default ratio cut:  R>0.98
    # Long energy cut is 162.4-10.9*abs(ieta)+0.21*ieta*ieta
    longEnergyParams        = cms.vdouble([43.5, 45.7, 48.32, 51.36,
                                           54.82, 58.7, 63.0, 67.72,
                                           72.86, 78.42, 84.4, 90.8,
                                           97.62]),
    longETParams            = cms.vdouble([0,0,0,0,
                                           0,0,0,0,
                                           0,0,0,0,0]),


    flagsToSkip             = cms.int32(string.atoi('0',2)), # HFDigiTime (bit 1) should be skipped, but not in MC
    short_R_29 = cms.vdouble([0.8]),
    long_R_29  = cms.vdouble([0.8]), # should move from 0.98 to 0.8?
    ),


                        # saturation and hfTimingTrust Parameters
                        saturationParameters=  cms.PSet(maxADCvalue=cms.int32(127)),

                        hfTimingTrustParameters = cms.PSet(hfTimingTrustLevel1=cms.int32(1), # 1ns timing accuracy
                                                           hfTimingTrustLevel2=cms.int32(4)  # 4ns timing accuracy
                                                           )

                        ) # cms.EDProducers


