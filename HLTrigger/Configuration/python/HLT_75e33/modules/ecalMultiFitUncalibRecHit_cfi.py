import FWCore.ParameterSet.Config as cms

ecalMultiFitUncalibRecHit = cms.EDProducer("EcalUncalibRecHitProducer",
    EBdigiCollection = cms.InputTag("ecalDigis","ebDigis"),
    EBhitCollection = cms.string('EcalUncalibRecHitsEB'),
    EEdigiCollection = cms.InputTag("ecalDigis","eeDigis"),
    EEhitCollection = cms.string('EcalUncalibRecHitsEE'),
    algo = cms.string('EcalUncalibRecHitWorkerMultiFit'),
    algoPSet = cms.PSet(
        EBamplitudeFitParameters = cms.vdouble(1.138, 1.652),
        EBtimeConstantTerm = cms.double(0.6),
        EBtimeFitLimits_Lower = cms.double(0.2),
        EBtimeFitLimits_Upper = cms.double(1.4),
        EBtimeFitParameters = cms.vdouble(
            -2.015452, 3.130702, -12.3473, 41.88921, -82.83944,
            91.01147, -50.35761, 11.05621
        ),
        EBtimeNconst = cms.double(28.5),
        EEamplitudeFitParameters = cms.vdouble(1.89, 1.4),
        EEtimeConstantTerm = cms.double(1.0),
        EEtimeFitLimits_Lower = cms.double(0.2),
        EEtimeFitLimits_Upper = cms.double(1.4),
        EEtimeFitParameters = cms.vdouble(
            -2.390548, 3.553628, -17.62341, 67.67538, -133.213,
            140.7432, -75.41106, 16.20277
        ),
        EEtimeNconst = cms.double(31.8),
        EcalPulseShapeParameters = cms.PSet(
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
            EBPulseShapeCovariance = cms.vdouble(
                3.001e-06, 1.233e-05, 0.0, -4.416e-06, -4.571e-06,
                -3.614e-06, -2.636e-06, -1.286e-06, -8.41e-07, -5.296e-07,
                0.0, 0.0, 1.233e-05, 6.154e-05, 0.0,
                -2.2e-05, -2.309e-05, -1.838e-05, -1.373e-05, -7.334e-06,
                -5.088e-06, -3.745e-06, -2.428e-06, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, -4.416e-06, -2.2e-05, 0.0, 8.319e-06,
                8.545e-06, 6.792e-06, 5.059e-06, 2.678e-06, 1.816e-06,
                1.223e-06, 8.245e-07, 5.589e-07, -4.571e-06, -2.309e-05,
                0.0, 8.545e-06, 9.182e-06, 7.219e-06, 5.388e-06,
                2.853e-06, 1.944e-06, 1.324e-06, 9.083e-07, 6.335e-07,
                -3.614e-06, -1.838e-05, 0.0, 6.792e-06, 7.219e-06,
                6.016e-06, 4.437e-06, 2.385e-06, 1.636e-06, 1.118e-06,
                7.754e-07, 5.556e-07, -2.636e-06, -1.373e-05, 0.0,
                5.059e-06, 5.388e-06, 4.437e-06, 3.602e-06, 1.917e-06,
                1.322e-06, 9.079e-07, 6.529e-07, 4.752e-07, -1.286e-06,
                -7.334e-06, 0.0, 2.678e-06, 2.853e-06, 2.385e-06,
                1.917e-06, 1.375e-06, 9.1e-07, 6.455e-07, 4.693e-07,
                3.657e-07, -8.41e-07, -5.088e-06, 0.0, 1.816e-06,
                1.944e-06, 1.636e-06, 1.322e-06, 9.1e-07, 9.115e-07,
                6.062e-07, 4.436e-07, 3.422e-07, -5.296e-07, -3.745e-06,
                0.0, 1.223e-06, 1.324e-06, 1.118e-06, 9.079e-07,
                6.455e-07, 6.062e-07, 7.217e-07, 4.862e-07, 3.768e-07,
                0.0, -2.428e-06, 0.0, 8.245e-07, 9.083e-07,
                7.754e-07, 6.529e-07, 4.693e-07, 4.436e-07, 4.862e-07,
                6.509e-07, 4.418e-07, 0.0, 0.0, 0.0,
                5.589e-07, 6.335e-07, 5.556e-07, 4.752e-07, 3.657e-07,
                3.422e-07, 3.768e-07, 4.418e-07, 6.142e-07
            ),
            EBPulseShapeTemplate = cms.vdouble(
                0.0113979, 0.758151, 1.0, 0.887744, 0.673548,
                0.474332, 0.319561, 0.215144, 0.147464, 0.101087,
                0.0693181, 0.0475044
            ),
            EBdigiCollection = cms.string(''),
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
            EEPulseShapeCovariance = cms.vdouble(
                3.941e-05, 3.333e-05, 0.0, -1.449e-05, -1.661e-05,
                -1.424e-05, -1.183e-05, -6.842e-06, -4.915e-06, -3.411e-06,
                0.0, 0.0, 3.333e-05, 2.862e-05, 0.0,
                -1.244e-05, -1.431e-05, -1.233e-05, -1.032e-05, -5.883e-06,
                -4.154e-06, -2.902e-06, -2.128e-06, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, -1.449e-05, -1.244e-05, 0.0, 5.84e-06,
                6.649e-06, 5.72e-06, 4.812e-06, 2.708e-06, 1.869e-06,
                1.33e-06, 9.186e-07, 6.446e-07, -1.661e-05, -1.431e-05,
                0.0, 6.649e-06, 7.966e-06, 6.898e-06, 5.794e-06,
                3.157e-06, 2.184e-06, 1.567e-06, 1.084e-06, 7.575e-07,
                -1.424e-05, -1.233e-05, 0.0, 5.72e-06, 6.898e-06,
                6.341e-06, 5.347e-06, 2.859e-06, 1.991e-06, 1.431e-06,
                9.839e-07, 6.886e-07, -1.183e-05, -1.032e-05, 0.0,
                4.812e-06, 5.794e-06, 5.347e-06, 4.854e-06, 2.628e-06,
                1.809e-06, 1.289e-06, 9.02e-07, 6.146e-07, -6.842e-06,
                -5.883e-06, 0.0, 2.708e-06, 3.157e-06, 2.859e-06,
                2.628e-06, 1.863e-06, 1.296e-06, 8.882e-07, 6.108e-07,
                4.283e-07, -4.915e-06, -4.154e-06, 0.0, 1.869e-06,
                2.184e-06, 1.991e-06, 1.809e-06, 1.296e-06, 1.217e-06,
                8.669e-07, 5.751e-07, 3.882e-07, -3.411e-06, -2.902e-06,
                0.0, 1.33e-06, 1.567e-06, 1.431e-06, 1.289e-06,
                8.882e-07, 8.669e-07, 9.522e-07, 6.717e-07, 4.293e-07,
                0.0, -2.128e-06, 0.0, 9.186e-07, 1.084e-06,
                9.839e-07, 9.02e-07, 6.108e-07, 5.751e-07, 6.717e-07,
                7.911e-07, 5.493e-07, 0.0, 0.0, 0.0,
                6.446e-07, 7.575e-07, 6.886e-07, 6.146e-07, 4.283e-07,
                3.882e-07, 4.293e-07, 5.493e-07, 7.027e-07
            ),
            EEPulseShapeTemplate = cms.vdouble(
                0.116442, 0.756246, 1.0, 0.897182, 0.686831,
                0.491506, 0.344111, 0.245731, 0.174115, 0.123361,
                0.0874288, 0.061957
            ),
            EEdigiCollection = cms.string(''),
            ESdigiCollection = cms.string(''),
            EcalPreMixStage1 = cms.bool(False),
            EcalPreMixStage2 = cms.bool(False),
            UseLCcorrection = cms.untracked.bool(True)
        ),
        activeBXs = cms.vint32(
            -5, -4, -3, -2, -1,
            0, 1, 2, 3, 4
        ),
        addPedestalUncertaintyEB = cms.double(0.0),
        addPedestalUncertaintyEE = cms.double(0.0),
        ampErrorCalculation = cms.bool(True),
        amplitudeThresholdEB = cms.double(10),
        amplitudeThresholdEE = cms.double(10),
        chi2ThreshEB_ = cms.double(65.0),
        chi2ThreshEE_ = cms.double(50.0),
        doPrefitEB = cms.bool(False),
        doPrefitEE = cms.bool(False),
        dynamicPedestalsEB = cms.bool(False),
        dynamicPedestalsEE = cms.bool(False),
        ebPulseShape = cms.vdouble(
            5.2e-05, -5.26e-05, 6.66e-05, 0.1168, 0.7575,
            1.0, 0.8876, 0.6732, 0.4741, 0.3194
        ),
        ebSpikeThreshold = cms.double(1.042),
        eePulseShape = cms.vdouble(
            5.2e-05, -5.26e-05, 6.66e-05, 0.1168, 0.7575,
            1.0, 0.8876, 0.6732, 0.4741, 0.3194
        ),
        gainSwitchUseMaxSampleEB = cms.bool(True),
        gainSwitchUseMaxSampleEE = cms.bool(False),
        kPoorRecoFlagEB = cms.bool(True),
        kPoorRecoFlagEE = cms.bool(False),
        mitigateBadSamplesEB = cms.bool(False),
        mitigateBadSamplesEE = cms.bool(False),
        outOfTimeThresholdGain12mEB = cms.double(5),
        outOfTimeThresholdGain12mEE = cms.double(1000),
        outOfTimeThresholdGain12pEB = cms.double(5),
        outOfTimeThresholdGain12pEE = cms.double(1000),
        outOfTimeThresholdGain61mEB = cms.double(5),
        outOfTimeThresholdGain61mEE = cms.double(1000),
        outOfTimeThresholdGain61pEB = cms.double(5),
        outOfTimeThresholdGain61pEE = cms.double(1000),
        prefitMaxChiSqEB = cms.double(25.0),
        prefitMaxChiSqEE = cms.double(10.0),
        selectiveBadSampleCriteriaEB = cms.bool(False),
        selectiveBadSampleCriteriaEE = cms.bool(False),
        simplifiedNoiseModelForGainSwitch = cms.bool(True),
        timealgo = cms.string('RatioMethod'),
        useLumiInfoRunHeader = cms.bool(True)
    )
)
