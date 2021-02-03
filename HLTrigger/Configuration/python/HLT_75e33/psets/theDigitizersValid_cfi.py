import FWCore.ParameterSet.Config as cms

theDigitizersValid = cms.PSet(
    calotruth = cms.PSet(
        HepMCProductLabel = cms.InputTag("generatorSmeared"),
        MaxPseudoRapidity = cms.double(5.0),
        MinEnergy = cms.double(0.5),
        accumulatorType = cms.string('CaloTruthAccumulator'),
        allowDifferentSimHitProcesses = cms.bool(False),
        doHGCAL = cms.bool(True),
        genParticleCollection = cms.InputTag("genParticles"),
        maximumPreviousBunchCrossing = cms.uint32(0),
        maximumSubsequentBunchCrossing = cms.uint32(0),
        premixStage1 = cms.bool(False),
        simHitCollections = cms.PSet(
            hgc = cms.VInputTag(cms.InputTag("g4SimHits","HGCHitsEE"), cms.InputTag("g4SimHits","HGCHitsHEfront"), cms.InputTag("g4SimHits","HGCHitsHEback"))
        ),
        simTrackCollection = cms.InputTag("g4SimHits"),
        simVertexCollection = cms.InputTag("g4SimHits")
    ),
    ecal = cms.PSet(
        ConstantTerm = cms.double(0.003),
        EBCorrNoiseMatrixG01 = cms.vdouble(
            1.0, 0.73354, 0.64442, 0.58851, 0.55425,
            0.53082, 0.51916, 0.51097, 0.50732, 0.50409
        ),
        EBCorrNoiseMatrixG06 = cms.vdouble(
            1.0, 0.70946, 0.58021, 0.49846, 0.45006,
            0.41366, 0.39699, 0.38478, 0.37847, 0.37055
        ),
        EBCorrNoiseMatrixG12 = cms.vdouble(
            1.0, 0.71073, 0.55721, 0.46089, 0.40449,
            0.35931, 0.33924, 0.32439, 0.31581, 0.30481
        ),
        EBdigiCollection = cms.string(''),
        EBs25notContainment = cms.double(0.9675),
        EECorrNoiseMatrixG01 = cms.vdouble(
            1.0, 0.72698, 0.62048, 0.55691, 0.51848,
            0.49147, 0.47813, 0.47007, 0.46621, 0.46265
        ),
        EECorrNoiseMatrixG06 = cms.vdouble(
            1.0, 0.71217, 0.47464, 0.34056, 0.26282,
            0.20287, 0.17734, 0.16256, 0.15618, 0.14443
        ),
        EECorrNoiseMatrixG12 = cms.vdouble(
            1.0, 0.71373, 0.44825, 0.30152, 0.21609,
            0.14786, 0.11772, 0.10165, 0.09465, 0.08098
        ),
        EEdigiCollection = cms.string(''),
        EEs25notContainment = cms.double(0.968),
        ESdigiCollection = cms.string(''),
        EcalPreMixStage1 = cms.bool(False),
        EcalPreMixStage2 = cms.bool(False),
        UseLCcorrection = cms.untracked.bool(True),
        accumulatorType = cms.string('EcalDigiProducer'),
        apdAddToBarrel = cms.bool(False),
        apdDigiTag = cms.string('APD'),
        apdDoPEStats = cms.bool(True),
        apdNonlParms = cms.vdouble(
            1.48, -3.75, 1.81, 1.26, 2.0,
            45, 1.0
        ),
        apdSeparateDigi = cms.bool(True),
        apdShapeTau = cms.double(40.5),
        apdShapeTstart = cms.double(74.5),
        apdSimToPEHigh = cms.double(88200000.0),
        apdSimToPELow = cms.double(2450000.0),
        apdTimeOffWidth = cms.double(0.8),
        apdTimeOffset = cms.double(-13.5),
        applyConstantTerm = cms.bool(True),
        binOfMaximum = cms.int32(6),
        cosmicsPhase = cms.bool(False),
        cosmicsShift = cms.double(0.0),
        doEB = cms.bool(True),
        doEE = cms.bool(False),
        doENoise = cms.bool(True),
        doES = cms.bool(False),
        doESNoise = cms.bool(True),
        doFast = cms.bool(True),
        doPhotostatistics = cms.bool(True),
        hitsProducer = cms.string('g4SimHits'),
        makeDigiSimLinks = cms.untracked.bool(False),
        photoelectronsToAnalogBarrel = cms.double(0.000444444),
        photoelectronsToAnalogEndcap = cms.double(0.000555555),
        readoutFrameSize = cms.int32(10),
        samplingFactor = cms.double(1.0),
        simHitToPhotoelectronsBarrel = cms.double(2250.0),
        simHitToPhotoelectronsEndcap = cms.double(1800.0),
        syncPhase = cms.bool(True),
        timeDependent = cms.bool(False),
        timePhase = cms.double(0.0)
    ),
    ecalTime = cms.PSet(
        EBtimeDigiCollection = cms.string('EBTimeDigi'),
        EEtimeDigiCollection = cms.string('EETimeDigi'),
        accumulatorType = cms.string('EcalTimeDigiProducer'),
        hitsProducerEB = cms.InputTag("g4SimHits","EcalHitsEB"),
        hitsProducerEE = cms.InputTag("g4SimHits","EcalHitsEE"),
        timeLayerBarrel = cms.int32(7),
        timeLayerEndcap = cms.int32(3)
    ),
    fastTimingLayer = cms.PSet(
        accumulatorType = cms.string('MTDDigiProducer'),
        barrelDigitizer = cms.PSet(
            DeviceSimulation = cms.PSet(
                LightCollectionEff = cms.double(0.25),
                LightCollectionSlopeL = cms.double(0.075),
                LightCollectionSlopeR = cms.double(0.075),
                LightYield = cms.double(40000.0),
                PhotonDetectionEff = cms.double(0.2),
                bxTime = cms.double(25)
            ),
            ElectronicsSimulation = cms.PSet(
                ChannelTimeOffset = cms.double(0.0),
                CorrelationCoefficient = cms.double(1.0),
                DarkCountRate = cms.double(10.0),
                EnergyThreshold = cms.double(4.0),
                Npe_to_V = cms.double(0.0064),
                Npe_to_pC = cms.double(0.016),
                ReferencePulseNpe = cms.double(100.0),
                ScintillatorDecayTime = cms.double(40.0),
                ScintillatorRiseTime = cms.double(1.1),
                SigmaClock = cms.double(0.015),
                SigmaElectronicNoise = cms.double(1.0),
                SinglePhotonTimeResolution = cms.double(0.06),
                SmearTimeForOOTtails = cms.bool(True),
                TestBeamMIPTimeRes = cms.double(4.293),
                TimeThreshold1 = cms.double(20.0),
                TimeThreshold2 = cms.double(50.0),
                adcNbits = cms.uint32(10),
                adcSaturation_MIP = cms.double(600.0),
                adcThreshold_MIP = cms.double(0.064),
                bxTime = cms.double(25),
                smearChannelTimeOffset = cms.double(0.0),
                tdcNbits = cms.uint32(10),
                toaLSB_ns = cms.double(0.02)
            ),
            digiCollectionTag = cms.string('FTLBarrel'),
            digitizerName = cms.string('BTLDigitizer'),
            inputSimHits = cms.InputTag("g4SimHits","FastTimerHitsBarrel"),
            maxSimHitsAccTime = cms.uint32(100),
            premixStage1 = cms.bool(False),
            premixStage1MaxCharge = cms.double(1000000.0),
            premixStage1MinCharge = cms.double(0.0001)
        ),
        endcapDigitizer = cms.PSet(
            DeviceSimulation = cms.PSet(
                bxTime = cms.double(25),
                meVPerMIP = cms.double(0.085),
                tofDelay = cms.double(1)
            ),
            ElectronicsSimulation = cms.PSet(
                adcNbits = cms.uint32(8),
                adcSaturation_MIP = cms.double(25),
                adcThreshold_MIP = cms.double(0.025),
                bxTime = cms.double(25),
                etaResolution = cms.string('0.03+0.0025*x'),
                tdcNbits = cms.uint32(11),
                toaLSB_ns = cms.double(0.013)
            ),
            digiCollectionTag = cms.string('FTLEndcap'),
            digitizerName = cms.string('ETLDigitizer'),
            inputSimHits = cms.InputTag("g4SimHits","FastTimerHitsEndcap"),
            maxSimHitsAccTime = cms.uint32(100),
            premixStage1 = cms.bool(False),
            premixStage1MaxCharge = cms.double(1000000.0),
            premixStage1MinCharge = cms.double(0.0001)
        ),
        makeDigiSimLinks = cms.bool(False),
        verbosity = cms.untracked.uint32(0)
    ),
    hcal = cms.PSet(
        DelivLuminosity = cms.double(0),
        HBDarkening = cms.bool(False),
        HEDarkening = cms.bool(False),
        HFDarkening = cms.bool(False),
        HFRecalParameterBlock = cms.PSet(
            HFdepthOneParameterA = cms.vdouble(
                0.004123, 0.00602, 0.008201, 0.010489, 0.013379,
                0.016997, 0.021464, 0.027371, 0.034195, 0.044807,
                0.058939, 0.125497
            ),
            HFdepthOneParameterB = cms.vdouble(
                -4e-06, -2e-06, 0.0, 4e-06, 1.5e-05,
                2.6e-05, 6.3e-05, 8.4e-05, 0.00016, 0.000107,
                0.000425, 0.000209
            ),
            HFdepthTwoParameterA = cms.vdouble(
                0.002861, 0.004168, 0.0064, 0.008388, 0.011601,
                0.014425, 0.018633, 0.023232, 0.028274, 0.035447,
                0.051579, 0.086593
            ),
            HFdepthTwoParameterB = cms.vdouble(
                -2e-06, -0.0, -7e-06, -6e-06, -2e-06,
                1e-06, 1.9e-05, 3.1e-05, 6.7e-05, 1.2e-05,
                0.000157, -3e-06
            )
        ),
        HcalPreMixStage1 = cms.bool(False),
        HcalPreMixStage2 = cms.bool(False),
        TestNumbering = cms.bool(True),
        accumulatorType = cms.string('HcalDigiProducer'),
        debugCaloSamples = cms.bool(False),
        doEmpty = cms.bool(True),
        doHFWindow = cms.bool(False),
        doIonFeedback = cms.bool(True),
        doNeutralDensityFilter = cms.bool(True),
        doNoise = cms.bool(True),
        doThermalNoise = cms.bool(True),
        doTimeSlew = cms.bool(True),
        hb = cms.PSet(
            binOfMaximum = cms.int32(6),
            doPhotoStatistics = cms.bool(True),
            doSiPMSmearing = cms.bool(True),
            firstRing = cms.int32(1),
            readoutFrameSize = cms.int32(10),
            samplingFactors = cms.vdouble(
                125.44, 125.54, 125.32, 125.13, 124.46,
                125.01, 125.22, 125.48, 124.45, 125.9,
                125.83, 127.01, 126.82, 129.73, 131.83,
                143.52
            ),
            simHitToPhotoelectrons = cms.double(2000.0),
            sipmTau = cms.double(10.0),
            syncPhase = cms.bool(True),
            threshold_currentTDC = cms.double(18.7),
            timePhase = cms.double(6.0),
            timeSmearing = cms.bool(True)
        ),
        he = cms.PSet(
            binOfMaximum = cms.int32(6),
            doPhotoStatistics = cms.bool(True),
            doSiPMSmearing = cms.bool(True),
            firstRing = cms.int32(16),
            readoutFrameSize = cms.int32(10),
            samplingFactors = cms.vdouble(
                210.55, 197.93, 186.12, 189.64, 189.63,
                189.96, 190.03, 190.11, 190.18, 190.25,
                190.32, 190.4, 190.47, 190.54, 190.61,
                190.69, 190.83, 190.94, 190.94, 190.94,
                190.94, 190.94, 190.94, 190.94, 190.94,
                190.94, 190.94, 190.94, 190.94, 190.94,
                190.94, 190.94, 190.94, 190.94, 190.94,
                190.94, 190.94, 190.94, 190.94, 190.94,
                190.94, 190.94, 190.94, 190.94, 190.94,
                190.94, 190.94, 190.94, 190.94, 190.94,
                190.94, 190.94, 190.94, 190.94, 190.94,
                190.94, 190.94, 190.94, 190.94, 190.94,
                190.94, 190.94, 190.94, 190.94, 190.94,
                190.94, 190.94, 190.94, 190.94, 190.94,
                190.94, 190.94, 190.94, 190.94, 190.94,
                190.94, 190.94, 190.94, 190.94, 190.94,
                190.94, 190.94, 190.94, 190.94, 190.94
            ),
            simHitToPhotoelectrons = cms.double(2000.0),
            sipmTau = cms.double(10.0),
            syncPhase = cms.bool(True),
            threshold_currentTDC = cms.double(18.7),
            timePhase = cms.double(6.0),
            timeSmearing = cms.bool(True)
        ),
        hf1 = cms.PSet(
            binOfMaximum = cms.int32(2),
            doPhotoStatistics = cms.bool(True),
            doSiPMSmearing = cms.bool(False),
            photoelectronsToAnalog = cms.double(2.79),
            readoutFrameSize = cms.int32(3),
            samplingFactor = cms.double(0.335),
            simHitToPhotoelectrons = cms.double(6.0),
            sipmTau = cms.double(0.0),
            syncPhase = cms.bool(True),
            threshold_currentTDC = cms.double(-999.0),
            timePhase = cms.double(14.0)
        ),
        hf2 = cms.PSet(
            binOfMaximum = cms.int32(2),
            doPhotoStatistics = cms.bool(True),
            doSiPMSmearing = cms.bool(False),
            photoelectronsToAnalog = cms.double(1.843),
            readoutFrameSize = cms.int32(3),
            samplingFactor = cms.double(0.335),
            simHitToPhotoelectrons = cms.double(6.0),
            sipmTau = cms.double(0.0),
            syncPhase = cms.bool(True),
            threshold_currentTDC = cms.double(-999.0),
            timePhase = cms.double(13.0)
        ),
        hitsProducer = cms.string('g4SimHits'),
        ho = cms.PSet(
            binOfMaximum = cms.int32(5),
            doPhotoStatistics = cms.bool(True),
            doSiPMSmearing = cms.bool(False),
            firstRing = cms.int32(1),
            readoutFrameSize = cms.int32(10),
            samplingFactors = cms.vdouble(
                231.0, 231.0, 231.0, 231.0, 360.0,
                360.0, 360.0, 360.0, 360.0, 360.0,
                360.0, 360.0, 360.0, 360.0, 360.0
            ),
            siPMCode = cms.int32(1),
            simHitToPhotoelectrons = cms.double(4000.0),
            sipmTau = cms.double(5.0),
            syncPhase = cms.bool(True),
            threshold_currentTDC = cms.double(-999.0),
            timePhase = cms.double(5.0),
            timeSmearing = cms.bool(False)
        ),
        hoHamamatsu = cms.PSet(
            binOfMaximum = cms.int32(5),
            doPhotoStatistics = cms.bool(True),
            doSiPMSmearing = cms.bool(False),
            firstRing = cms.int32(1),
            readoutFrameSize = cms.int32(10),
            samplingFactors = cms.vdouble(
                231.0, 231.0, 231.0, 231.0, 360.0,
                360.0, 360.0, 360.0, 360.0, 360.0,
                360.0, 360.0, 360.0, 360.0, 360.0
            ),
            siPMCode = cms.int32(2),
            simHitToPhotoelectrons = cms.double(4000.0),
            sipmTau = cms.double(5.0),
            syncPhase = cms.bool(True),
            threshold_currentTDC = cms.double(-999.0),
            timePhase = cms.double(5.0),
            timeSmearing = cms.bool(False)
        ),
        hoZecotek = cms.PSet(
            binOfMaximum = cms.int32(5),
            doPhotoStatistics = cms.bool(True),
            doSiPMSmearing = cms.bool(False),
            firstRing = cms.int32(1),
            readoutFrameSize = cms.int32(10),
            samplingFactors = cms.vdouble(
                231.0, 231.0, 231.0, 231.0, 360.0,
                360.0, 360.0, 360.0, 360.0, 360.0,
                360.0, 360.0, 360.0, 360.0, 360.0
            ),
            siPMCode = cms.int32(2),
            simHitToPhotoelectrons = cms.double(4000.0),
            sipmTau = cms.double(5.0),
            syncPhase = cms.bool(True),
            threshold_currentTDC = cms.double(-999.0),
            timePhase = cms.double(5.0),
            timeSmearing = cms.bool(False)
        ),
        ignoreGeantTime = cms.bool(False),
        injectTestHits = cms.bool(False),
        injectTestHitsCells = cms.vint32(),
        injectTestHitsEnergy = cms.vdouble(),
        injectTestHitsTime = cms.vdouble(),
        killHE = cms.bool(True),
        makeDigiSimLinks = cms.untracked.bool(False),
        minFCToDelay = cms.double(5.0),
        zdc = cms.PSet(
            binOfMaximum = cms.int32(5),
            doPhotoStatistics = cms.bool(True),
            doSiPMSmearing = cms.bool(False),
            photoelectronsToAnalog = cms.double(1.843),
            readoutFrameSize = cms.int32(10),
            samplingFactor = cms.double(1.0),
            simHitToPhotoelectrons = cms.double(6.0),
            sipmTau = cms.double(0.0),
            syncPhase = cms.bool(True),
            threshold_currentTDC = cms.double(-999.0),
            timePhase = cms.double(-4.0)
        )
    ),
    hgceeDigitizer = cms.PSet(
        NoiseGeneration_Method = cms.bool(True),
        accumulatorType = cms.string('HGCDigiProducer'),
        bxTime = cms.double(25),
        digiCfg = cms.PSet(
            cceParams = cms.PSet(
                refToPSet_ = cms.string('HGCAL_cceParams_toUse')
            ),
            chargeCollectionEfficiencies = cms.PSet(
                refToPSet_ = cms.string('HGCAL_chargeCollectionEfficiencies')
            ),
            doTimeSamples = cms.bool(False),
            feCfg = cms.PSet(
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
            ),
            ileakParam = cms.PSet(
                refToPSet_ = cms.string('HGCAL_ileakParam_toUse')
            ),
            keV2fC = cms.double(0.044259),
            noise_fC = cms.PSet(
                refToPSet_ = cms.string('HGCAL_noise_fC')
            ),
            thresholdFollowsMIP = cms.bool(True)
        ),
        digiCollection = cms.string('HGCDigisEE'),
        digitizationType = cms.uint32(0),
        eVPerEleHolePair = cms.double(3.62),
        geometryType = cms.uint32(1),
        hitCollection = cms.string('HGCHitsEE'),
        makeDigiSimLinks = cms.bool(False),
        maxSimHitsAccTime = cms.uint32(100),
        premixStage1 = cms.bool(False),
        premixStage1MaxCharge = cms.double(1000000.0),
        premixStage1MinCharge = cms.double(0),
        tofDelay = cms.double(5),
        useAllChannels = cms.bool(True),
        verbosity = cms.untracked.uint32(0)
    ),
    hgchebackDigitizer = cms.PSet(
        NoiseGeneration_Method = cms.bool(True),
        accumulatorType = cms.string('HGCDigiProducer'),
        bxTime = cms.double(25),
        digiCfg = cms.PSet(
            algo = cms.uint32(2),
            doTimeSamples = cms.bool(False),
            feCfg = cms.PSet(
                adcNbits = cms.uint32(10),
                adcPulse = cms.vdouble(
                    0.0, 0.017, 0.817, 0.163, 0.003,
                    0.0
                ),
                adcSaturation_fC = cms.double(68.75),
                adcThreshold_fC = cms.double(0.5),
                fwVersion = cms.uint32(2),
                jitterConstant_ns = cms.vdouble(0.0004, 0.0004, 0.0004),
                jitterNoise_ns = cms.vdouble(25.0, 25.0, 25.0),
                pulseAvgT = cms.vdouble(
                    0.0, 23.42298, 13.16733, 6.41062, 5.03946,
                    4.532
                ),
                targetMIPvalue_ADC = cms.uint32(15),
                tdcChargeDrainParameterisation = cms.vdouble(
                    -919.13, 365.36, -14.1, 0.2, -21.85,
                    49.39, 22.21, 0.8, -0.28, 27.14,
                    43.95, 3.89048
                ),
                tdcForToAOnset_fC = cms.vdouble(12.0, 12.0, 12.0),
                tdcNbits = cms.uint32(12),
                tdcOnset_fC = cms.double(55),
                tdcResolutionInPs = cms.double(0.001),
                tdcSaturation_fC = cms.double(1000),
                toaLSB_ns = cms.double(0.0244),
                toaMode = cms.uint32(1)
            ),
            keV2MIP = cms.double(0.00148148148148),
            nPEperMIP = cms.double(21.0),
            nTotalPE = cms.double(7500),
            noise = cms.PSet(
                refToPSet_ = cms.string('HGCAL_noise_heback')
            ),
            scaleBySipmArea = cms.bool(False),
            scaleByTileArea = cms.bool(False),
            sdPixels = cms.double(1e-06),
            sipmMap = cms.string('SimCalorimetry/HGCalSimProducers/data/sipmParams_geom-10.txt'),
            thresholdFollowsMIP = cms.bool(True),
            xTalk = cms.double(0.01)
        ),
        digiCollection = cms.string('HGCDigisHEback'),
        digitizationType = cms.uint32(1),
        geometryType = cms.uint32(1),
        hitCollection = cms.string('HGCHitsHEback'),
        makeDigiSimLinks = cms.bool(False),
        maxSimHitsAccTime = cms.uint32(100),
        premixStage1 = cms.bool(False),
        premixStage1MaxCharge = cms.double(1000000.0),
        premixStage1MinCharge = cms.double(0),
        tofDelay = cms.double(1),
        useAllChannels = cms.bool(True),
        verbosity = cms.untracked.uint32(0)
    ),
    hgchefrontDigitizer = cms.PSet(
        NoiseGeneration_Method = cms.bool(True),
        accumulatorType = cms.string('HGCDigiProducer'),
        bxTime = cms.double(25),
        digiCfg = cms.PSet(
            cceParams = cms.PSet(
                refToPSet_ = cms.string('HGCAL_cceParams_toUse')
            ),
            chargeCollectionEfficiencies = cms.PSet(
                refToPSet_ = cms.string('HGCAL_chargeCollectionEfficiencies')
            ),
            doTimeSamples = cms.bool(False),
            feCfg = cms.PSet(
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
            ),
            ileakParam = cms.PSet(
                refToPSet_ = cms.string('HGCAL_ileakParam_toUse')
            ),
            keV2fC = cms.double(0.044259),
            noise_fC = cms.PSet(
                refToPSet_ = cms.string('HGCAL_noise_fC')
            ),
            thresholdFollowsMIP = cms.bool(True)
        ),
        digiCollection = cms.string('HGCDigisHEfront'),
        digitizationType = cms.uint32(0),
        geometryType = cms.uint32(1),
        hitCollection = cms.string('HGCHitsHEfront'),
        makeDigiSimLinks = cms.bool(False),
        maxSimHitsAccTime = cms.uint32(100),
        premixStage1 = cms.bool(False),
        premixStage1MaxCharge = cms.double(1000000.0),
        premixStage1MinCharge = cms.double(0),
        tofDelay = cms.double(5),
        useAllChannels = cms.bool(True),
        verbosity = cms.untracked.uint32(0)
    ),
    mergedtruth = cms.PSet(
        HepMCProductLabel = cms.InputTag("generatorSmeared"),
        accumulatorType = cms.string('TrackingTruthAccumulator'),
        allowDifferentSimHitProcesses = cms.bool(False),
        alwaysAddAncestors = cms.bool(True),
        createInitialVertexCollection = cms.bool(True),
        createMergedBremsstrahlung = cms.bool(True),
        createUnmergedCollection = cms.bool(True),
        genParticleCollection = cms.InputTag("genParticles"),
        ignoreTracksOutsideVolume = cms.bool(False),
        maximumPreviousBunchCrossing = cms.uint32(9999),
        maximumSubsequentBunchCrossing = cms.uint32(9999),
        removeDeadModules = cms.bool(False),
        select = cms.PSet(
            chargedOnlyTP = cms.bool(True),
            intimeOnlyTP = cms.bool(False),
            lipTP = cms.double(1000),
            maxRapidityTP = cms.double(5.0),
            minHitTP = cms.int32(0),
            minRapidityTP = cms.double(-5.0),
            pdgIdTP = cms.vint32(),
            ptMaxTP = cms.double(1e+100),
            ptMinTP = cms.double(0.1),
            signalOnlyTP = cms.bool(False),
            stableOnlyTP = cms.bool(False),
            tipTP = cms.double(1000)
        ),
        simHitCollections = cms.PSet(
            muon = cms.VInputTag(cms.InputTag("g4SimHits","MuonDTHits"), cms.InputTag("g4SimHits","MuonCSCHits"), cms.InputTag("g4SimHits","MuonRPCHits"), cms.InputTag("g4SimHits","MuonGEMHits"), cms.InputTag("g4SimHits","MuonME0Hits")),
            pixel = cms.VInputTag(cms.InputTag("g4SimHits","TrackerHitsPixelBarrelLowTof"), cms.InputTag("g4SimHits","TrackerHitsPixelBarrelHighTof"), cms.InputTag("g4SimHits","TrackerHitsPixelEndcapLowTof"), cms.InputTag("g4SimHits","TrackerHitsPixelEndcapHighTof")),
            tracker = cms.VInputTag()
        ),
        simTrackCollection = cms.InputTag("g4SimHits"),
        simVertexCollection = cms.InputTag("g4SimHits"),
        vertexDistanceCut = cms.double(0.003),
        volumeRadius = cms.double(120.0),
        volumeZ = cms.double(300.0)
    ),
    pixel = cms.PSet(
        AlgorithmCommon = cms.PSet(
            DeltaProductionCut = cms.double(0.03),
            makeDigiSimLinks = cms.untracked.bool(True)
        ),
        GeometryType = cms.string('idealForDigi'),
        PSPDigitizerAlgorithm = cms.PSet(
            AdcFullScale = cms.int32(255),
            AddInefficiency = cms.bool(False),
            AddNoise = cms.bool(True),
            AddNoisyPixels = cms.bool(True),
            AddThresholdSmearing = cms.bool(True),
            AddXTalk = cms.bool(True),
            Alpha2Order = cms.bool(True),
            CellsToKill = cms.VPSet(),
            ClusterWidth = cms.double(3),
            DeadModules = cms.VPSet(),
            DeadModules_DB = cms.bool(False),
            EfficiencyFactors_Barrel = cms.vdouble(
                0.999, 0.999, 0.999, 0.999, 0.999,
                0.999, 0.999, 0.999, 0.999, 0.999
            ),
            EfficiencyFactors_Endcap = cms.vdouble(
                0.999, 0.999, 0.999, 0.999, 0.999,
                0.999, 0.999, 0.999, 0.999, 0.999,
                0.999, 0.999, 0.999, 0.999, 0.999,
                0.999
            ),
            ElectronPerAdc = cms.double(135.0),
            HIPThresholdInElectrons_Barrel = cms.double(10000000000.0),
            HIPThresholdInElectrons_Endcap = cms.double(10000000000.0),
            Inefficiency_DB = cms.bool(False),
            InterstripCoupling = cms.double(0.05),
            KillModules = cms.bool(False),
            LorentzAngle_DB = cms.bool(False),
            NoiseInElectrons = cms.double(200),
            Phase2ReadoutMode = cms.int32(0),
            ReadoutNoiseInElec = cms.double(200.0),
            SigmaCoeff = cms.double(1.8),
            SigmaZero = cms.double(0.00037),
            TanLorentzAnglePerTesla_Barrel = cms.double(0.07),
            TanLorentzAnglePerTesla_Endcap = cms.double(0.07),
            ThresholdInElectrons_Barrel = cms.double(6300.0),
            ThresholdInElectrons_Endcap = cms.double(6300.0),
            ThresholdSmearing_Barrel = cms.double(630.0),
            ThresholdSmearing_Endcap = cms.double(630.0),
            TofLowerCut = cms.double(-12.5),
            TofUpperCut = cms.double(12.5)
        ),
        PSSDigitizerAlgorithm = cms.PSet(
            AdcFullScale = cms.int32(255),
            AddInefficiency = cms.bool(False),
            AddNoise = cms.bool(True),
            AddNoisyPixels = cms.bool(True),
            AddThresholdSmearing = cms.bool(True),
            AddXTalk = cms.bool(True),
            Alpha2Order = cms.bool(True),
            CellsToKill = cms.VPSet(),
            ClusterWidth = cms.double(3),
            DeadModules = cms.VPSet(),
            DeadModules_DB = cms.bool(False),
            EfficiencyFactors_Barrel = cms.vdouble(
                0.999, 0.999, 0.999, 0.999, 0.999,
                0.999, 0.999, 0.999, 0.999, 0.999
            ),
            EfficiencyFactors_Endcap = cms.vdouble(
                0.999, 0.999, 0.999, 0.999, 0.999,
                0.999, 0.999, 0.999, 0.999, 0.999,
                0.999, 0.999, 0.999, 0.999, 0.999,
                0.999
            ),
            ElectronPerAdc = cms.double(135.0),
            HIPThresholdInElectrons_Barrel = cms.double(21000.0),
            HIPThresholdInElectrons_Endcap = cms.double(21000.0),
            Inefficiency_DB = cms.bool(False),
            InterstripCoupling = cms.double(0.05),
            KillModules = cms.bool(False),
            LorentzAngle_DB = cms.bool(False),
            NoiseInElectrons = cms.double(700),
            Phase2ReadoutMode = cms.int32(0),
            ReadoutNoiseInElec = cms.double(700.0),
            SigmaCoeff = cms.double(1.8),
            SigmaZero = cms.double(0.00037),
            TanLorentzAnglePerTesla_Barrel = cms.double(0.07),
            TanLorentzAnglePerTesla_Endcap = cms.double(0.07),
            ThresholdInElectrons_Barrel = cms.double(6300.0),
            ThresholdInElectrons_Endcap = cms.double(6300.0),
            ThresholdSmearing_Barrel = cms.double(630.0),
            ThresholdSmearing_Endcap = cms.double(630.0),
            TofLowerCut = cms.double(-12.5),
            TofUpperCut = cms.double(12.5)
        ),
        Pixel3DDigitizerAlgorithm = cms.PSet(
            AdcFullScale = cms.int32(15),
            AddInefficiency = cms.bool(False),
            AddNoise = cms.bool(False),
            AddNoisyPixels = cms.bool(False),
            AddThresholdSmearing = cms.bool(False),
            AddXTalk = cms.bool(False),
            Alpha2Order = cms.bool(True),
            CellsToKill = cms.VPSet(),
            ClusterWidth = cms.double(3),
            DeadModules = cms.VPSet(),
            DeadModules_DB = cms.bool(False),
            EfficiencyFactors_Barrel = cms.vdouble(
                0.999, 0.999, 0.999, 0.999, 0.999,
                0.999, 0.999, 0.999, 0.999, 0.999
            ),
            EfficiencyFactors_Endcap = cms.vdouble(
                0.999, 0.999, 0.999, 0.999, 0.999,
                0.999, 0.999, 0.999, 0.999, 0.999,
                0.999, 0.999, 0.999, 0.999, 0.999,
                0.999
            ),
            ElectronPerAdc = cms.double(600.0),
            Even_column_interchannelCoupling_next_column = cms.double(0.0),
            Even_row_interchannelCoupling_next_row = cms.double(0.0),
            HIPThresholdInElectrons_Barrel = cms.double(10000000000.0),
            HIPThresholdInElectrons_Endcap = cms.double(10000000000.0),
            Inefficiency_DB = cms.bool(False),
            InterstripCoupling = cms.double(0.0),
            KillModules = cms.bool(False),
            LorentzAngle_DB = cms.bool(True),
            NoiseInElectrons = cms.double(0.0),
            Odd_column_interchannelCoupling_next_column = cms.double(0.0),
            Odd_row_interchannelCoupling_next_row = cms.double(0.2),
            Phase2ReadoutMode = cms.int32(-1),
            ReadoutNoiseInElec = cms.double(0.0),
            SigmaCoeff = cms.double(1.8),
            SigmaZero = cms.double(0.00037),
            TanLorentzAnglePerTesla_Barrel = cms.double(0.106),
            TanLorentzAnglePerTesla_Endcap = cms.double(0.106),
            ThresholdInElectrons_Barrel = cms.double(1200.0),
            ThresholdInElectrons_Endcap = cms.double(1200.0),
            ThresholdSmearing_Barrel = cms.double(0.0),
            ThresholdSmearing_Endcap = cms.double(0.0),
            TofLowerCut = cms.double(-12.5),
            TofUpperCut = cms.double(12.5)
        ),
        PixelDigitizerAlgorithm = cms.PSet(
            AdcFullScale = cms.int32(15),
            AddInefficiency = cms.bool(False),
            AddNoise = cms.bool(False),
            AddNoisyPixels = cms.bool(False),
            AddThresholdSmearing = cms.bool(False),
            AddXTalk = cms.bool(False),
            Alpha2Order = cms.bool(True),
            ApplyTimewalk = cms.bool(False),
            CellsToKill = cms.VPSet(),
            ClusterWidth = cms.double(3),
            DeadModules = cms.VPSet(),
            DeadModules_DB = cms.bool(False),
            EfficiencyFactors_Barrel = cms.vdouble(
                0.999, 0.999, 0.999, 0.999, 0.999,
                0.999, 0.999, 0.999, 0.999, 0.999
            ),
            EfficiencyFactors_Endcap = cms.vdouble(
                0.999, 0.999, 0.999, 0.999, 0.999,
                0.999, 0.999, 0.999, 0.999, 0.999,
                0.999, 0.999, 0.999, 0.999, 0.999,
                0.999
            ),
            ElectronPerAdc = cms.double(600.0),
            Even_column_interchannelCoupling_next_column = cms.double(0.0),
            Even_row_interchannelCoupling_next_row = cms.double(0.0),
            HIPThresholdInElectrons_Barrel = cms.double(10000000000.0),
            HIPThresholdInElectrons_Endcap = cms.double(10000000000.0),
            Inefficiency_DB = cms.bool(False),
            InterstripCoupling = cms.double(0.0),
            KillModules = cms.bool(False),
            LorentzAngle_DB = cms.bool(True),
            NoiseInElectrons = cms.double(0.0),
            Odd_column_interchannelCoupling_next_column = cms.double(0.0),
            Odd_row_interchannelCoupling_next_row = cms.double(0.2),
            Phase2ReadoutMode = cms.int32(-1),
            ReadoutNoiseInElec = cms.double(0.0),
            SigmaCoeff = cms.double(0),
            SigmaZero = cms.double(0.00037),
            TanLorentzAnglePerTesla_Barrel = cms.double(0.106),
            TanLorentzAnglePerTesla_Endcap = cms.double(0.106),
            ThresholdInElectrons_Barrel = cms.double(1200.0),
            ThresholdInElectrons_Endcap = cms.double(1200.0),
            ThresholdSmearing_Barrel = cms.double(0.0),
            ThresholdSmearing_Endcap = cms.double(0.0),
            TimewalkModel = cms.PSet(
                Curves = cms.VPSet(
                    cms.PSet(
                        charge = cms.vdouble(
                            1000, 1025, 1050, 1100, 1200,
                            1500, 2000, 6000, 10000, 15000,
                            20000, 30000
                        ),
                        delay = cms.vdouble(
                            26.8, 23.73, 21.92, 19.46, 16.52,
                            12.15, 8.88, 3.03, 1.69, 0.95,
                            0.56, 0.19
                        )
                    ),
                    cms.PSet(
                        charge = cms.vdouble(
                            1200, 1225, 1250, 1500, 2000,
                            6000, 10000, 15000, 20000, 30000
                        ),
                        delay = cms.vdouble(
                            26.28, 23.5, 21.79, 14.92, 10.27,
                            3.33, 1.86, 1.07, 0.66, 0.27
                        )
                    ),
                    cms.PSet(
                        charge = cms.vdouble(
                            1500, 1525, 1550, 1600, 2000,
                            6000, 10000, 15000, 20000, 30000
                        ),
                        delay = cms.vdouble(
                            25.36, 23.05, 21.6, 19.56, 12.94,
                            3.79, 2.14, 1.26, 0.81, 0.39
                        )
                    ),
                    cms.PSet(
                        charge = cms.vdouble(
                            3000, 3025, 3050, 3100, 3500,
                            6000, 10000, 15000, 20000, 30000
                        ),
                        delay = cms.vdouble(
                            25.63, 23.63, 22.35, 20.65, 14.92,
                            6.7, 3.68, 2.29, 1.62, 1.02
                        )
                    )
                ),
                ThresholdValues = cms.vdouble(1000, 1200, 1500, 3000)
            ),
            TofLowerCut = cms.double(-12.5),
            TofUpperCut = cms.double(12.5)
        ),
        ROUList = cms.vstring(
            'TrackerHitsPixelBarrelLowTof',
            'TrackerHitsPixelBarrelHighTof',
            'TrackerHitsPixelEndcapLowTof',
            'TrackerHitsPixelEndcapHighTof'
        ),
        SSDigitizerAlgorithm = cms.PSet(
            AdcFullScale = cms.int32(255),
            AddInefficiency = cms.bool(False),
            AddNoise = cms.bool(True),
            AddNoisyPixels = cms.bool(True),
            AddThresholdSmearing = cms.bool(True),
            AddXTalk = cms.bool(True),
            Alpha2Order = cms.bool(True),
            CBCDeadTime = cms.double(0.0),
            CellsToKill = cms.VPSet(),
            ClusterWidth = cms.double(3),
            DeadModules = cms.VPSet(),
            DeadModules_DB = cms.bool(False),
            EfficiencyFactors_Barrel = cms.vdouble(
                0.999, 0.999, 0.999, 0.999, 0.999,
                0.999, 0.999, 0.999, 0.999, 0.999
            ),
            EfficiencyFactors_Endcap = cms.vdouble(
                0.999, 0.999, 0.999, 0.999, 0.999,
                0.999, 0.999, 0.999, 0.999, 0.999,
                0.999, 0.999, 0.999, 0.999, 0.999,
                0.999
            ),
            ElectronPerAdc = cms.double(135.0),
            HIPThresholdInElectrons_Barrel = cms.double(10000000000.0),
            HIPThresholdInElectrons_Endcap = cms.double(10000000000.0),
            HitDetectionMode = cms.int32(0),
            Inefficiency_DB = cms.bool(False),
            InterstripCoupling = cms.double(0.05),
            KillModules = cms.bool(False),
            LorentzAngle_DB = cms.bool(False),
            NoiseInElectrons = cms.double(1000),
            Phase2ReadoutMode = cms.int32(0),
            PulseShapeParameters = cms.vdouble(
                -3.0, 16.043703, 99.999857, 40.57165, 2.0,
                1.2459094
            ),
            ReadoutNoiseInElec = cms.double(1000.0),
            SigmaCoeff = cms.double(1.8),
            SigmaZero = cms.double(0.00037),
            TanLorentzAnglePerTesla_Barrel = cms.double(0.07),
            TanLorentzAnglePerTesla_Endcap = cms.double(0.07),
            ThresholdInElectrons_Barrel = cms.double(5800.0),
            ThresholdInElectrons_Endcap = cms.double(5800.0),
            ThresholdSmearing_Barrel = cms.double(580.0),
            ThresholdSmearing_Endcap = cms.double(580.0),
            TofLowerCut = cms.double(-12.5),
            TofUpperCut = cms.double(12.5)
        ),
        accumulatorType = cms.string('Phase2TrackerDigitizer'),
        hitsProducer = cms.string('g4SimHits'),
        isOTreadoutAnalog = cms.bool(False),
        premixStage1 = cms.bool(False)
    ),
    puVtx = cms.PSet(
        accumulatorType = cms.string('PileupVertexAccumulator'),
        hitsProducer = cms.string('generator'),
        makeDigiSimLinks = cms.untracked.bool(False),
        saveVtxTimes = cms.bool(True),
        vtxFallbackTag = cms.InputTag("generator"),
        vtxTag = cms.InputTag("generatorSmeared")
    ),
    strip = cms.PSet(
        APVProbabilityFile = cms.FileInPath('SimTracker/SiStripDigitizer/data/APVProbaList.txt'),
        APVSaturationFromHIP = cms.bool(False),
        APVSaturationProbScaling = cms.double(1.0),
        APVShapeDecoFile = cms.FileInPath('SimTracker/SiStripDigitizer/data/APVShapeDeco_320.txt'),
        APVShapePeakFile = cms.FileInPath('SimTracker/SiStripDigitizer/data/APVShapePeak_default.txt'),
        APVpeakmode = cms.bool(False),
        AppliedVoltage = cms.double(300.0),
        BaselineShift = cms.bool(True),
        ChargeDistributionRMS = cms.double(6.5e-10),
        ChargeMobility = cms.double(310.0),
        CommonModeNoise = cms.bool(True),
        CosmicDelayShift = cms.untracked.double(0.0),
        CouplingConstantDecIB1 = cms.vdouble(0.7748, 0.0962, 0.0165),
        CouplingConstantDecIB2 = cms.vdouble(0.83, 0.0756, 0.0094),
        CouplingConstantDecOB1 = cms.vdouble(0.6871, 0.1222, 0.0342),
        CouplingConstantDecOB2 = cms.vdouble(0.725, 0.1102, 0.0273),
        CouplingConstantDecW1a = cms.vdouble(0.786, 0.093, 0.014),
        CouplingConstantDecW1b = cms.vdouble(0.822, 0.08, 0.009),
        CouplingConstantDecW2a = cms.vdouble(0.7962, 0.0914, 0.0104),
        CouplingConstantDecW2b = cms.vdouble(0.888, 0.05, 0.006),
        CouplingConstantDecW3a = cms.vdouble(0.8164, 0.09, 0.0018),
        CouplingConstantDecW3b = cms.vdouble(0.848, 0.06, 0.016),
        CouplingConstantDecW4 = cms.vdouble(0.876, 0.06, 0.002),
        CouplingConstantDecW5 = cms.vdouble(0.7565, 0.0913, 0.0304),
        CouplingConstantDecW6 = cms.vdouble(0.758, 0.093, 0.026),
        CouplingConstantDecW7 = cms.vdouble(0.7828, 0.0862, 0.0224),
        CouplingConstantPeakIB1 = cms.vdouble(0.9006, 0.0497),
        CouplingConstantPeakIB2 = cms.vdouble(0.9342, 0.0328),
        CouplingConstantPeakOB1 = cms.vdouble(0.8542, 0.0729),
        CouplingConstantPeakOB2 = cms.vdouble(0.8719, 0.064),
        CouplingConstantPeakW1a = cms.vdouble(0.996, 0.002),
        CouplingConstantPeakW1b = cms.vdouble(0.976, 0.012),
        CouplingConstantPeakW2a = cms.vdouble(1.0, 0.0),
        CouplingConstantPeakW2b = cms.vdouble(0.998, 0.001),
        CouplingConstantPeakW3a = cms.vdouble(0.996, 0.002),
        CouplingConstantPeakW3b = cms.vdouble(0.992, 0.004),
        CouplingConstantPeakW4 = cms.vdouble(0.992, 0.004),
        CouplingConstantPeakW5 = cms.vdouble(0.968, 0.016),
        CouplingConstantPeakW6 = cms.vdouble(0.972, 0.014),
        CouplingConstantPeakW7 = cms.vdouble(0.964, 0.018),
        CouplingConstantRunIIDecIB1 = cms.vdouble(0.8361, 0.0703, 0.0117),
        CouplingConstantRunIIDecIB2 = cms.vdouble(0.8616, 0.0588, 0.0104),
        CouplingConstantRunIIDecOB1 = cms.vdouble(0.7461, 0.0996, 0.0273),
        CouplingConstantRunIIDecOB2 = cms.vdouble(0.7925, 0.0834, 0.0203),
        CouplingConstantRunIIDecW1a = cms.vdouble(0.8571, 0.0608, 0.0106),
        CouplingConstantRunIIDecW1b = cms.vdouble(0.8827, 0.0518, 0.0068),
        CouplingConstantRunIIDecW2a = cms.vdouble(0.8861, 0.049, 0.008),
        CouplingConstantRunIIDecW2b = cms.vdouble(0.8943, 0.0483, 0.0046),
        CouplingConstantRunIIDecW3a = cms.vdouble(0.8984, 0.0494, 0.0014),
        CouplingConstantRunIIDecW3b = cms.vdouble(0.8611, 0.0573, 0.0121),
        CouplingConstantRunIIDecW4 = cms.vdouble(0.8881, 0.0544, 0.0015),
        CouplingConstantRunIIDecW5 = cms.vdouble(0.7997, 0.077, 0.0231),
        CouplingConstantRunIIDecW6 = cms.vdouble(0.8067, 0.0769, 0.0198),
        CouplingConstantRunIIDecW7 = cms.vdouble(0.7883, 0.0888, 0.0171),
        CouplingConstantsRunIIDecB = cms.bool(True),
        CouplingConstantsRunIIDecW = cms.bool(True),
        DeltaProductionCut = cms.double(0.120425),
        DepletionVoltage = cms.double(170.0),
        DigiModeList = cms.PSet(
            PRDigi = cms.string('ProcessedRaw'),
            SCDigi = cms.string('ScopeMode'),
            VRDigi = cms.string('VirginRaw'),
            ZSDigi = cms.string('ZeroSuppressed')
        ),
        FedAlgorithm = cms.int32(4),
        FedAlgorithm_PM = cms.int32(4),
        Gain = cms.string(''),
        GeometryType = cms.string('idealForDigi'),
        GevPerElectron = cms.double(3.61e-09),
        Inefficiency = cms.double(0.0),
        LandauFluctuations = cms.bool(True),
        LorentzAngle = cms.string(''),
        Noise = cms.bool(True),
        NoiseSigmaThreshold = cms.double(2.0),
        PedestalsOffset = cms.double(128),
        PreMixingMode = cms.bool(False),
        ROUList = cms.vstring(
            'g4SimHitsTrackerHitsPixelBarrelLowTof',
            'g4SimHitsTrackerHitsPixelEndcapLowTof'
        ),
        RealPedestals = cms.bool(True),
        SingleStripNoise = cms.bool(True),
        TOFCutForDeconvolution = cms.double(50.0),
        TOFCutForPeak = cms.double(100.0),
        Temperature = cms.double(273.0),
        TrackerConfigurationFromDB = cms.bool(False),
        ZeroSuppression = cms.bool(True),
        accumulatorType = cms.string('SiStripDigitizer'),
        apv_mVPerQ = cms.double(5.5),
        apv_maxResponse = cms.double(729),
        apv_rate = cms.double(66.2),
        apvfCPerElectron = cms.double(0.0001602),
        chargeDivisionsPerStrip = cms.int32(10),
        cmnRMStec = cms.double(2.44),
        cmnRMStib = cms.double(5.92),
        cmnRMStid = cms.double(3.08),
        cmnRMStob = cms.double(1.08),
        electronPerAdcDec = cms.double(247.0),
        electronPerAdcPeak = cms.double(262.0),
        fracOfEventsToSimAPV = cms.double(0.0),
        hitsProducer = cms.string('g4SimHits'),
        includeAPVSimulation = cms.bool(False),
        makeDigiSimLinks = cms.untracked.bool(True),
        noDiffusion = cms.bool(False)
    )
)