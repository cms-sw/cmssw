import FWCore.ParameterSet.Config as cms

EcalTrivialConditionRetriever = cms.ESSource("EcalTrivialConditionRetriever",
    producedEcalChannelStatus = cms.untracked.bool(True),
    producedEcalDCSTowerStatus = cms.untracked.bool(True),
    producedEcalDAQTowerStatus = cms.untracked.bool(True),
    producedEcalTrgChannelStatus = cms.untracked.bool(True),
    #       Values to get correct noise on RecHit amplitude using 3+5 weights
    # new 2010 March 26 to get correlation into account
    EBpedRMSX12 = cms.untracked.double(1.089),
    weightsForTB = cms.untracked.bool(False),
    # channel status
    channelStatusFile = cms.untracked.string(''),
    producedEcalPedestals = cms.untracked.bool(True),
    #       If set true reading optimized weights (3+5 weights) from file 
    getWeightsFromFile = cms.untracked.bool(True),
    intercalibErrorsFile = cms.untracked.string(''),
    laserAPDPNMean = cms.untracked.double(1.0),
    laserAPDPNRefMean = cms.untracked.double(1.0),
    #       untracked string amplWeightsFile = "CalibCalorimetry/EcalTrivialCondModules/data/ampWeights_TB.txt"
    # file with intercalib constants - same format used for online and offline DB
    # by default set all inter calib const to 1.0 if no file provided
    intercalibConstantsFile = cms.untracked.string(''),
    producedEcalWeights = cms.untracked.bool(True),
    # new 2010 March 26 to get correlation into account
    EEpedRMSX12 = cms.untracked.double(2.018),
    producedEcalIntercalibConstants = cms.untracked.bool(True),
    producedEcalIntercalibConstantsMC = cms.untracked.bool(True),
    producedEcalIntercalibErrors = cms.untracked.bool(True),
    producedEcalTimeCalibConstants = cms.untracked.bool(True),
    producedEcalTimeCalibErrors = cms.untracked.bool(True),
    producedEcalLaserCorrection = cms.untracked.bool(True),
    producedEcalGainRatios = cms.untracked.bool(True),
    producedEcalADCToGeVConstant = cms.untracked.bool(True),
    adcToGeVEBConstant = cms.untracked.double(0.035),
    adcToGeVEEConstant = cms.untracked.double(0.06),
    # cluster functions/corrections -- by default no parameters are passed
    producedEcalClusterLocalContCorrParameters = cms.untracked.bool(True),
    localContCorrParameters = cms.untracked.vdouble( 
            1.00365, 0.0007179, -0.008303, 0.01116, -0.1057, 1.00362, 0.0006617, -0.005505, -0.01044, -0.1770, 1.0035),
    producedEcalClusterCrackCorrParameters = cms.untracked.bool(True),
    crackCorrParameters = cms.untracked.vdouble( 
            0.9933, -0.01813, -0.03359, -0.09972, -0.2889, 0.9909, 0.04019, 
            -0.1095, 0.2401, -0.3412, 0.9942, -0.01245, -0.03002, -0.1098, 
            -0.2777, 0.9981, 0.01087, -0.01359, 0.06212, -0.354),
    mappingFile = cms.untracked.string('Geometry/EcalMapping/data/EEMap.txt'),
    producedEcalMappingElectronics = cms.untracked.bool(True),
    energyUncertaintyParameters = cms.untracked.vdouble(
            0.002793, 0.000908,  0.23592,   0.04446,
            0.02463, -0.001782, -0.343492, -0.017968,
            -0.013338, 0.0013819, 0.398369,  0.025488,
            0.002264, 0.000674,  0.281829,  0.043100,
            0.02047, -0.001914, -0.297824, -0.020220,
            -0.010669, 0.001648,  0.464209, -0.01112,
            0.000530, 0.001274,  0.21071,   0.04679,
            0.031323, -0.001997, -0.40509,  -0.05102,
            -0.016961, 0.0014051, 0.313083,  0.059649,
            -0.03947,  0.03364,   3.6768,    0.243637,
            0.05167, -0.02335,  -2.77506,  -0.162785,
            -0.011482, 0.004621,  0.511206,  0.032607,
            -0.05062,  0.057102,  5.48885,  -0.5305,
            0.06604,  -0.04686,  -4.34245,   0.500381,
            -0.01487,  0.010382,  0.823244, -0.09392,
            -0.04195,  0.028296,  1.66651,   0.87535,
            0.048104, -0.01493,  -0.98163,  -0.72297,
            -0.010256, 0.001827,  0.149991,  0.144294),
    producedEnergyUncertaintyParameters = cms.untracked.bool(True),
    energyCorrectionParameters = cms.untracked.vdouble(
            40.2198, -3.03103e-6,
            1.1, 8.0, -0.05185, 0.1354, 0.9165, -0.0005626, 1.385,
            1.002,  -0.7424, 0,            0,
            0,        0.5558,  2.375,   0.1869,
            7.6,      1.081,  -0.00181,
            0, 0,
            0.9, 6.5, -0.1214, 0.2362, 0.8847, -0.00193, 1.057,
            2.213, -17.29,
            -0.599,  8.874,
            0.09632, -1.457,
            -0.7584,  10.29,
            1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 0, 0, 0, 0, 0, 0, 0),
    produceEnergyCorrectionParameters = cms.untracked.bool(True)
)
# foo bar baz
