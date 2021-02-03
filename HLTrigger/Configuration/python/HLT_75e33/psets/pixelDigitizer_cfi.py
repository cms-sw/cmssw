import FWCore.ParameterSet.Config as cms

pixelDigitizer = cms.PSet(
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
)