import FWCore.ParameterSet.Config as cms

hfreco = cms.EDProducer("HcalHitReconstructor",
                        correctionPhaseNS = cms.double(0.0),
                        digiLabel = cms.InputTag("hcalDigis"),
                        samplesToAdd = cms.int32(1),
                        Subdetector = cms.string('HF'),
                        firstSample = cms.int32(3),
                        correctForPhaseContainment = cms.bool(False),
                        correctForTimeslew = cms.bool(False),
                        dropZSmarkedPassed = cms.bool(True),
                        
                        # Tags for calculating status flags
                        correctTiming = cms.bool(True),
                        setNoiseFlags = cms.bool(True),

                        # HF Noise algorithm choices:
                        #  1 = flat energy/ET cut; flag channel if R=|(L-S)/(L+S)| is greater than a fixed threshold;
                        #  2 = PET algorithm:  still flag if R> threshold, but allow energy/ET cuts to be parameterized functions of ieta;
                        #  3 = default algorithm:  Apply PET for long fibers and those at |ieta|=29, but use S9S1 for other short fibers
                        #  4 = S9S1-only algorithm:  Use S9S1 test everywhere, even for short fibers.  (as of March 2010, short fiber parameters for S9S1 have not been tested)
                        
                        HFNoiseAlgo   = cms.int32(3), # Default algorithm should be algo 3 (PET+S9S1 combo)
                        
                        setHSCPFlags  = cms.bool(True),
                        setSaturationFlags = cms.bool(True),
                        setTimingTrustFlags = cms.bool(True),
                        
                        digistat= cms.PSet(HFpulsetimemin     = cms.int32(0),
                                           HFpulsetimemax     = cms.int32(10), # min/max time slice values for peak
                                           HFratio_beforepeak = cms.double(0.6), # max allowed ratio (started at 0.1, loosened to 0.6 after pion studies)
                                           HFratio_afterpeak  = cms.double(1.0), # max allowed ratio
                                           HFadcthreshold       = cms.int32(10), # minimum size of peak (in ADC counts, after ped subtraction) to be considered noisy
                                           ),
                        
                        rechitstat = cms.PSet(short_HFlongshortratio = cms.double(0.995), # max allowed ratio of (L-S)/(L+S)
                                            short_HFETthreshold    = cms.double(0.), # minimum ET (in GeV) required for a cell to be considered hot (started at 0.5, loosened to 2.0 after pion studies)
                                            short_HFEnergythreshold      = cms.double(50), # minimum energy (in GeV) required for a cell to be considered hot
                                            
                                            long_HFlongshortratio = cms.double(0.995), # max allowed ratio of (L-S)/(L+S)
                                            long_HFETthreshold    = cms.double(0.), # minimum ET (in GeV) required for a cell to be considered hot (started at 0.5, loosened to 2.0 after pion studies)
                                            long_HFEnergythreshold      = cms.double(50), # minimum energy (in GeV) required for a cell to be considered hot
                                            ), # rechitstat

                        # Parameters for Using S9S1 Test
                        #     optimumSlopeParams are the coefficients for the slope function m = [0]+[1]*(ieta)+[2]*(ieta)^2...
                        #     Slopes are given fixed values at ieta=40 and ieta=41
                        #     energy and ET params are coefficients for energy/ET thresholds, parameterized in ieta
                        S9S1stat = cms.PSet( long_optimumSlopeParams = cms.vdouble([0.3084,-0.02577, 0.0005351]),
                                             long_optimumSlope40     = cms.double(0.0913756),
                                             long_optimumSlope41     = cms.double(0.0589927),
                                             longEnergyParams        = cms.vdouble([162.4,-10.19,0.21]),
                                             longETParams            = cms.vdouble([0]),
                                             # WARNING!  SHORT SLOPE PARAMETERS ARE NOT USED IN ANY OF THE AVAILABLE DEFAULT ALGORITHMS!
                                             # DEFAULT ALGO 3 USES PET RATIO TEST
                                             short_optimumSlopeParams = cms.vdouble([0.3084,-0.02577, 0.0005351]),
                                             short_optimumSlope40     = cms.double(0.0913756),
                                             short_optimumSlope41     = cms.double(0.0589927),
                                             shortEnergyParams        = cms.vdouble([129.9,-6.61,0.1153]),
                                             shortETParams            = cms.vdouble([0]),
                                            ),

                        # Parameters for Using Parameterized Energy Threshold (PET) test
                        #  short_R, long_R are coefficients of R threshold, parameterized in *ENERGY*:  R_thresh = [0]+[1]*energy+[2]*energy^2+...
                        #  As of March 2010, the R threshold is a simply fixed value:  R>0.98
                        #  Energy and ET params are energy and ET threshold coefficients, parameterized in *ieta*
                        
                        PETstat = cms.PSet(short_R = cms.vdouble([0.98]),  # default ratio cut:  R>0.98
                                           shortEnergyParams        = cms.vdouble([129.9,-6.61,0.1153]),
                                           shortETParams            = cms.vdouble([0]),  # by default, only trivial cut ET>0 applied
                                           long_R  = cms.vdouble([0.98]),  # default ratio cut:  R>0.98
                                           longEnergyParams        = cms.vdouble([162.4,-10.19,0.21]),
                                           longETParams            = cms.vdouble([0])    # by default, only trivial cut ET>0 applied
                                           ),

                        
                        saturationParameters=  cms.PSet(maxADCvalue=cms.int32(127)),
                        hfTimingTrustParameters = cms.PSet(hfTimingTrustLevel1=cms.int32(1), # 1ns timing accuracy
                                                           hfTimingTrustLevel2=cms.int32(4)  # 4ns timing accuracy
                                                           )
                        
                        ) # cms.EDProducers


