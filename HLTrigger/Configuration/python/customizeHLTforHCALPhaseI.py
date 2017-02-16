import FWCore.ParameterSet.Config as cms

import itertools

# customisation functions for the HLT configuration
from HLTrigger.Configuration.common import *

# import the relevant eras from Configuration.Eras.*
from Configuration.Eras.Modifier_run2_HCAL_2017_cff import run2_HCAL_2017
from Configuration.Eras.Modifier_run2_HF_2017_cff import run2_HF_2017


# modify the HLT configuration for the Phase I HE upgrade
def customizeHLTforHEforPhaseI(process):

    # reconstruct HBHE rechits with Method 3
    hltHbhereco = cms.EDProducer( "HBHEPhase1Reconstructor",

        # Label for the input HBHEDigiCollection, and flag indicating
        # whether we should process this collection
        digiLabelQIE8 = cms.InputTag("hltHcalDigis"),
        processQIE8 = cms.bool(True),

        # Label for the input QIE11DigiCollection, and flag indicating
        # whether we should process this collection
        digiLabelQIE11 = cms.InputTag("hltHcalDigis"),
        processQIE11 = cms.bool(True),

        # Get the "sample of interest" index from DB?
        # If not, it is taken from the dataframe.
        tsFromDB = cms.bool(False),

        # Use the HcalRecoParam structure from DB inside
        # the reconstruction algorithm?
        recoParamsFromDB = cms.bool(True),

        # Drop zero-suppressed channels?
        dropZSmarkedPassed = cms.bool(True),

        # Flag indicating whether we should produce HBHERecHitCollection
        makeRecHits = cms.bool(True),

        # Flag indicating whether we should produce HBHEChannelInfoCollection
        saveInfos = cms.bool(False),

        # Flag indicating whether we should include HBHEChannelInfo objects
        # into HBHEChannelInfoCollection despite the fact that the channels
        # are either tagged bad in DB of zero-suppressed. Note that the rechit
        # collection will not include such channels even if this flag is set.
        saveDroppedInfos = cms.bool(False),

        # Configure the reconstruction algorithm
        algorithm = cms.PSet(
            Class = cms.string("SimpleHBHEPhase1Algo"),

            # Time shift (in ns) to add to TDC timing (for QIE11)
            tdcTimeShift = cms.double(0.),

            # Parameters for "Method 0"
            firstSampleShift            = cms.int32(0),
            samplesToAdd                = cms.int32(2),
            correctForPhaseContainment  = cms.bool(True),
            correctionPhaseNS           = cms.double(6.),

            # Parameters for Method 2
            useM2 = cms.bool(False),
            applyPedConstraint  = cms.bool(True),
            applyTimeConstraint = cms.bool(True),
            applyPulseJitter    = cms.bool(False),
            applyTimeSlew       = cms.bool(True),               # units
            ts4Min              = cms.double(0.),               # fC
            ts4Max              = cms.vdouble(100., 45000.),     # fC # this is roughly 20 GeV
            pulseJitter         = cms.double(1.),               # GeV/bin 
            meanTime            = cms.double(0.),               # ns 
            timeSigmaHPD        = cms.double(5.),               # ns 
            timeSigmaSiPM       = cms.double(2.5),              # ns
            meanPed             = cms.double(0.),               # GeV
            pedSigmaHPD         = cms.double(0.5),              # GeV
            pedSigmaSiPM        = cms.double(0.00065),          # GeV - this correspond roughtly to 1.5 fC for a gain of 2276
            noiseHPD            = cms.double(1),                # fC
            noiseSiPM           = cms.double(1),                # fC
            timeMin             = cms.double(-12.5),            # ns
            timeMax             = cms.double(12.5),             # ns
            ts4chi2             = cms.vdouble(15., 15.),         # chi2 for triple pulse
            fitTimes            = cms.int32(1),                 # -1 means no constraint on number of fits per channel

            # Parameters for Method 3
            useM3 = cms.bool(True),
            applyTimeSlewM3     = cms.bool(True),
            pedestalUpperLimit  = cms.double(2.7),
            timeSlewParsType    = cms.int32(3),                 # 0: TestStand, 1:Data, 2:MC, 3:InputPars. Parametrization function is par0 + par1*log(fC+par2).
            timeSlewPars        = cms.vdouble(12.2999, -2.19142, 0, 12.2999, -2.19142, 0, 12.2999, -2.19142, 0), 
                                                                # HB par0, HB par1, HB par2, BE par0, BE par1, BE par2, HE par0, HE par1, HE par2
            respCorrM3          = cms.double(0.95)              # This factor is used to align the the Method3 with the Method2 response
        ),

        # Reconstruction algorithm configuration data to fetch from DB, if any
        algoConfigClass = cms.string(""),

        # Turn rechit status bit setters on/off
        setNegativeFlags        = cms.bool(False),
        setNoiseFlagsQIE8       = cms.bool(True),
        setNoiseFlagsQIE11      = cms.bool(False),
        setPulseShapeFlagsQIE8  = cms.bool(True),
        setPulseShapeFlagsQIE11 = cms.bool(False),
        setLegacyFlagsQIE8      = cms.bool(True),
        setLegacyFlagsQIE11     = cms.bool(False),

        # Parameter sets configuring rechit status bit setters for HPD
        flagParametersQIE8 = cms.PSet(
            nominalPedestal = cms.double(3.),                   # fC
            hitEnergyMinimum = cms.double(1.),                  # GeV
            hitMultiplicityThreshold = cms.int32(17),
            pulseShapeParameterSets = cms.VPSet(
                cms.PSet( pulseShapeParameters = cms.vdouble(   0.0, 100.0, -50.0, 0.0, -15.0, 0.15) ),
                cms.PSet( pulseShapeParameters = cms.vdouble( 100.0, 2.0e3, -50.0, 0.0,  -5.0, 0.05) ),
                cms.PSet( pulseShapeParameters = cms.vdouble( 2.0e3, 1.0e6, -50.0, 0.0,  95.0, 0.0 ) ),
                cms.PSet( pulseShapeParameters = cms.vdouble(-1.0e6, 1.0e6,  45.0, 0.1, 1.0e6, 0.0 ) ),
            )
        ),

        # Pulse shape parametrisation for HPD
        pulseShapeParametersQIE8 = cms.PSet(
            MinimumChargeThreshold      = cms.double(20),
            TS4TS5ChargeThreshold       = cms.double(70),
            TS3TS4ChargeThreshold       = cms.double(70),
            TS3TS4UpperChargeThreshold  = cms.double(20),
            TS5TS6ChargeThreshold       = cms.double(70),
            TS5TS6UpperChargeThreshold  = cms.double(20),
            R45PlusOneRange             = cms.double(0.2),
            R45MinusOneRange            = cms.double(0.2),
            TrianglePeakTS              = cms.uint32(10000),    # Disable the "triangle peak fit" and the corresponding HBHETriangleNoise flag
            TriangleIgnoreSlow          = cms.bool(False),
            LinearThreshold             = cms.vdouble(20, 100, 100000),
            LinearCut                   = cms.vdouble(-3, -0.054, -0.054),
            RMS8MaxThreshold            = cms.vdouble(20, 100, 100000),
            RMS8MaxCut                  = cms.vdouble(-13.5, -11.5, -11.5),
            LeftSlopeThreshold          = cms.vdouble(250, 500, 100000),
            LeftSlopeCut                = cms.vdouble(5, 2.55, 2.55),
            RightSlopeThreshold         = cms.vdouble(250, 400, 100000),
            RightSlopeCut               = cms.vdouble(5, 4.15, 4.15),
            RightSlopeSmallThreshold    = cms.vdouble(150, 200, 100000),
            RightSlopeSmallCut          = cms.vdouble(1.08, 1.16, 1.16),
            MinimumTS4TS5Threshold      = cms.double(100),
            TS4TS5UpperThreshold        = cms.vdouble(70, 90, 100, 400),
            TS4TS5UpperCut              = cms.vdouble(1, 0.8, 0.75, 0.72),
            TS4TS5LowerThreshold        = cms.vdouble(100, 120, 160, 200, 300, 500),
            TS4TS5LowerCut              = cms.vdouble(-1, -0.7, -0.5, -0.4, -0.3, 0.1),
            UseDualFit                  = cms.bool(True),
        ),

        # Pulse shape parametrisation for SiPM
        pulseShapeParametersQIE11 = cms.PSet( ),

        # Parameter sets configuring rechit status bit setters for SiPM
        flagParametersQIE11 = cms.PSet( )
    )

    # XXX these values were used at HLT in 2016, but we do not know why
    #hltHbhereco.algorithm.samplesToAdd      = 4
    #hltHbhereco.algorithm.correctionPhaseNS = 13.
    #hltHbhereco.setNoiseFlagsQIE8           = False
    #hltHbhereco.setPulseShapeFlagsQIE8      = False
    #hltHbhereco.setLegacyFlagsQIE8          = False
   
    # reconstruct HBHE rechits with Method 3
    if 'hltHbhereco' in process.__dict__:
        digiLabel = process.hltHbhereco.digiLabel.value()
        process.hltHbhereco = hltHbhereco.clone()
        process.hltHbhereco.digiLabelQIE8  = digiLabel
        process.hltHbhereco.digiLabelQIE11 = digiLabel

    # reconstruct HBHE rechits with Method 2 around E/Gamma candidates (seeded by L1 objects)
    if 'hltHbherecoMethod2L1EGSeeded' in process.__dict__:
        digiLabel = process.hltHbherecoMethod2L1EGSeeded.digiLabel.value()
        process.hltHbherecoMethod2L1EGSeeded = hltHbhereco.clone()
        process.hltHbherecoMethod2L1EGSeeded.digiLabelQIE8   = digiLabel
        # set processQIE11 to False until HLTHcalDigisInRegionsProducer can produce QIE11
        process.hltHbherecoMethod2L1EGSeeded.processQIE11    = cms.bool(False)
        process.hltHbherecoMethod2L1EGSeeded.digiLabelQIE11  = cms.InputTag('')
        process.hltHbherecoMethod2L1EGSeeded.algorithm.useM2 = cms.bool(True)
        process.hltHbherecoMethod2L1EGSeeded.algorithm.useM3 = cms.bool(False)

    # reconstruct HBHE rechits with Method 2 around E/Gamma candidates (unseeded)
    if 'hltHbherecoMethod2L1EGUnseeded' in process.__dict__:
        digiLabel = process.hltHbherecoMethod2L1EGUnseeded.digiLabel.value()
        process.hltHbherecoMethod2L1EGUnseeded = hltHbhereco.clone()
        process.hltHbherecoMethod2L1EGUnseeded.digiLabelQIE8   = digiLabel
        # set processQIE11 to False until HLTHcalDigisInRegionsProducer can produce QIE11
        process.hltHbherecoMethod2L1EGUnseeded.processQIE11    = cms.bool(False)
        process.hltHbherecoMethod2L1EGUnseeded.digiLabelQIE11  = cms.InputTag('')
        process.hltHbherecoMethod2L1EGUnseeded.algorithm.useM2 = cms.bool(True)
        process.hltHbherecoMethod2L1EGUnseeded.algorithm.useM3 = cms.bool(False)

    return process


# attach `customizeHLTforHEforPhaseI' to the `run2_HCAL_2017' era
def modifyHLTforHEforPhaseI(process):
    run2_HCAL_2017.toModify(process, customizeHLTforHEforPhaseI)


# modify the HLT configuration for the Phase I HF upgrade
def customizeHLTforHFforPhaseI(process):

    if 'hltHfreco' in process.__dict__:

        process.hltHfprereco = cms.EDProducer("HFPreReconstructor",
            digiLabel = cms.InputTag("hltHcalDigis"),
            dropZSmarkedPassed = cms.bool(True),
            tsFromDB = cms.bool(False),
            sumAllTimeSlices = cms.bool(False)
        )

        process.hltHfreco = cms.EDProducer("HFPhase1Reconstructor",
            # Label for the input HFPreRecHitCollection
            inputLabel = cms.InputTag("hltHfprereco"),

            # Change the following to True in order to use the channel
            # status from the DB
            useChannelQualityFromDB = cms.bool(False),

            # Change the following to True when the status becomes
            # available in the DB for both anodes. If this parameter
            # is set to False then it is assumed that the status of
            # both anodes is given by the channel at depth 1 and 2.
            checkChannelQualityForDepth3and4 = cms.bool(False),

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

                # Do not construct rechits with problems
                rejectAllFailures = cms.bool(True)
            ),

            # Reconstruction algorithm data to fetch from DB, if any
            algoConfigClass = cms.string("HFPhase1PMTParams"),

            # Turn on/off the noise cleanup algorithms
            setNoiseFlags = cms.bool(False),

            # Parameters for the S9S1 test.
            #
            #   optimumSlopes are slopes for each of the |ieta| values
            #   29, 30, .... , 41  (although |ieta|=29 is not used in
            #   current S9S1 formulation)
            #
            #   energy and ET params are thresholds for each |ieta|
            #
            S9S1stat = cms.PSet(
                # WARNING!  ONLY LONG PARAMETERS ARE USED IN DEFAULT RECO; SHORT S9S1 IS NOT USED!
                short_optimumSlope      = cms.vdouble( -99999, 0.0164905, 0.0238698, 0.0321383, 0.041296, 0.0513428, 0.0622789, 0.0741041, 0.0868186, 0.100422, 0.135313, 0.136289, 0.0589927 ),
                # Short energy cut is 129.9 - 6.61*|ieta|+0.1153*|ieta|^2
                shortEnergyParams       = cms.vdouble( 35.1773, 35.37, 35.7933, 36.4472, 37.3317, 38.4468, 39.7925, 41.3688, 43.1757, 45.2132, 47.4813, 49.98, 52.7093 ),
                shortETParams           = cms.vdouble( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ),
                long_optimumSlope       = cms.vdouble( -99999, 0.0164905, 0.0238698, 0.0321383, 0.041296, 0.0513428, 0.0622789, 0.0741041, 0.0868186, 0.100422, 0.135313, 0.136289, 0.0589927 ),
                # Long energy cut is 162.4-10.9*abs(ieta)+0.21*ieta*ieta
                longEnergyParams        = cms.vdouble( 43.5, 45.7, 48.32, 51.36, 54.82, 58.7, 63.0, 67.72, 72.86, 78.42, 84.4, 90.8, 97.62 ),
                longETParams            = cms.vdouble( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ),
                HcalAcceptSeverityLevel = cms.int32(9),         # allow hits with severity up to AND INCLUDING 9
                isS8S1                  = cms.bool(False),
            ),

            # Parameters for the S8S1 test. Sets the HFS8S1Ratio Bit (bit 3).
            #
            #   energy and ET params are coefficients for
            #   energy/ET thresholds, parameterized in ieta
            #
            S8S1stat = cms.PSet(
                # ieta=29 is a special case
                short_optimumSlope      = cms.vdouble( 0.30, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10 ),
                # Short energy cut is 40 for ieta=29, 100 otherwise
                shortEnergyParams       = cms.vdouble( 40, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100 ),
                shortETParams           = cms.vdouble( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ),
                # ieta=29 is a special case
                long_optimumSlope       = cms.vdouble( 0.30, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10 ),
                # Long energy cut is 40 for ieta=29, 100 otherwise
                longEnergyParams        = cms.vdouble( 40, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100 ),
                longETParams            = cms.vdouble( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ),
                HcalAcceptSeverityLevel = cms.int32(9),         # allow hits with severity up to AND INCLUDING 9
                isS8S1                  = cms.bool(True),
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
                short_R                 = cms.vdouble( 0.8 ),   # new default ratio cut:  R>0.8
                short_R_29              = cms.vdouble( 0.8 ),
                # Short energy cut is 129.9 - 6.61*|ieta|+0.1153*|ieta|^2
                shortEnergyParams       = cms.vdouble( 35.1773, 35.37, 35.7933, 36.4472, 37.3317, 38.4468, 39.7925, 41.3688, 43.1757, 45.2132, 47.4813, 49.98, 52.7093 ),
                shortETParams           = cms.vdouble( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ),
                long_R                  = cms.vdouble( 0.98 ),  # default ratio cut:  R>0.98
                long_R_29               = cms.vdouble( 0.8 ),   # should move from 0.98 to 0.8?
                # Long energy cut is 162.4-10.9*abs(ieta)+0.21*ieta*ieta
                longEnergyParams        = cms.vdouble( 43.5, 45.7, 48.32, 51.36,54.82, 58.7, 63.0, 67.72, 72.86, 78.42, 84.4, 90.8, 97.62 ),
                longETParams            = cms.vdouble( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ),
                HcalAcceptSeverityLevel = cms.int32(9),         # allow hits with severity up to AND INCLUDING 9
            )
        )

        # add the hltHfprereco module before the hltHfreco in any Sequence, Paths or EndPath that contains the latter
        for sequence in itertools.chain(
            process._Process__sequences.itervalues(),
            process._Process__paths.itervalues(),
            process._Process__endpaths.itervalues()
        ):
            try:
                position = sequence.index(process.hltHfreco)
            except ValueError:
                continue
            else:
                sequence.insert(position, process.hltHfprereco)

    return process


# attach `customizeHLTforHFforPhaseI' to the `run2_HF_2017' era
def modifyHLTforHFforPhaseI(process):
    run2_HF_2017.toModify(process, customizeHLTforHFforPhaseI)

