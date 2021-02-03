import FWCore.ParameterSet.Config as cms

mtdDigitizer = cms.PSet(
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
)