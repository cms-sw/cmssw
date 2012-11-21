import FWCore.ParameterSet.Config as cms

EcalTrivialConditionRetriever = cms.ESSource("EcalTrivialConditionRetriever",
    producedEcalChannelStatus = cms.untracked.bool(True),
    producedEcalDQMTowerStatus = cms.untracked.bool(True),
    producedEcalDQMChannelStatus = cms.untracked.bool(True),
    producedEcalDCSTowerStatus = cms.untracked.bool(True),
    producedEcalDAQTowerStatus = cms.untracked.bool(True),
    producedEcalTrgChannelStatus = cms.untracked.bool(True),
    #       Values to get correct noise on RecHit amplitude using 3+5 weights
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
    linearCorrectionsFile = cms.untracked.string(''),
    producedEcalWeights = cms.untracked.bool(True),
    EEpedRMSX12 = cms.untracked.double(2.018),
    producedEcalLinearCorrections = cms.untracked.bool(True),
    producedEcalIntercalibConstants = cms.untracked.bool(True),
    producedEcalIntercalibConstantsMC = cms.untracked.bool(True),
    producedEcalIntercalibErrors = cms.untracked.bool(True),
    producedEcalTimeCalibConstants = cms.untracked.bool(True),
    producedEcalTimeCalibErrors = cms.untracked.bool(True),
    producedEcalTimeOffsetConstant = cms.untracked.bool(True),
    producedEcalLaserCorrection = cms.untracked.bool(True),
    producedEcalGainRatios = cms.untracked.bool(True),
    producedEcalADCToGeVConstant = cms.untracked.bool(True),
    adcToGeVEBConstant = cms.untracked.double(0.035),
    adcToGeVEEConstant = cms.untracked.double(0.06),
    # cluster functions/corrections -- by default no parameters are passed
    producedEcalClusterLocalContCorrParameters = cms.untracked.bool(True),
    localContCorrParameters = cms.untracked.vdouble( 
#            1.00365, 0.0007179, -0.008303, 0.01116, -0.1057, 1.00362, 0.0006617, -0.005505, -0.01044, -0.1770, 1.0035),
# Monte Carlo (Apr 2012)
#    1.00385, 0.000847402, 0.0419403, 1.0033,  0.00100782,  0.0362918,
#    1.00322, 0.000902587, 0.0335483, 1.00478, 0.000112104, 0.05377,
#    1.00363, -0.00168853, 0.0392934, 1.00374, -0.00197705, 0.0402998,
#    1.00258, -0.00121254, 0.0278283, 1.00266, 0.00165111,  0.0245362),
# data (Apr 2012)
    1.00603, 0.00300789,  0.0667232, 1.00655, 0.00386189,  0.073931,
    1.00634, 0.00631341,  0.0764134, 1.00957, 0.0113306,   0.123808,
    1.00403, -0.0012733,  0.042925,  1.00394, -0.00137567, 0.0416698,
    1.00298, -0.00111589, 0.0320377, 1.00269, -0.00153347, 0.0296769),
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
    producedEcalClusterEnergyUncertaintyParameters = cms.untracked.bool(True),
    energyCorrectionParameters = cms.untracked.vdouble(
#            40.2198, -3.03103e-6,
#            1.1, 8.0, -0.05185, 0.1354, 0.9165, -0.0005626, 1.385,
#            1.002,  -0.7424, 0,            0,
#            0,        0.5558,  2.375,   0.1869,
#            7.6,      1.081,  -0.00181,
#            0, 0,
#            0.9, 6.5, -0.1214, 0.2362, 0.8847, -0.00193, 1.057,
#            2.213, -17.29,
#            -0.599,  8.874,
#            0.09632, -1.457,
#            -0.7584,  10.29,
#            1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#            1, 0, 0, 0, 0, 0, 0, 0),
#   New dat from Yurii Maravin (2011/03/02)
             40.2198, -3.03103e-6,
             1.1, 8.0, -0.05289, 0.1374, 0.9141, -0.000669, 1.38,
             1.000,  -0.698, 0,            0,
             0,        0.6605,  8.825,   0.841,
             7.6,      1.081,  -0.00181,
             0, 0,
             0.9, 6.5, -0.07945, 0.1298, 0.9147, -0.001565, 0.9,
             -3.516, -2.362,
             2.151, 1.572,
             -0.336, -0.2807,
             3.2,  0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0),
    producedEcalClusterEnergyCorrectionParameters = cms.untracked.bool(True),
    energyCorrectionObjectSpecificParameters = cms.untracked.vdouble(
    # 2011 Nov 17
# fEta : p0, p1
#             40.2198, -3.03103e-6,
## fBremEta : xcorr,par0, par1, par2, par3, par4 (x 14 x 2 (electron/photon))
# Electrons
#xcorr
#             1.00227, 1.00252, 1.00225, 1.00159, 0.999475, 0.997203, 0.993886,
#             0.971262, 0.975922, 0.979087, 0.98495, 0.98781, 0.989546, 0.989638,
#par
#             1.00718, -0.00187886, 0, 0, 0,
#             1.00713, -0.00227574, 0, 0, 0,
#             1.00641, -0.00259935, 0, 0, 0,
#             1.00761, -0.00433692, 0, 0, 0,
#             1.00682, -0.00551324, 0, 0, 0,
#             1.0073, -0.00799669, 0, 0, 0,
#             1.00462, -0.00870057, 0, 0, 0,
#             0.972798, -0.000771577, -0.00276696, 0, 0,
#             0.981672, -0.00202028, -0.00471028, 0, 0,
#             0.98251, 0.00441308, -0.00809139, 0, 0,
#             0.986123, 0.00832913, -0.00944584, 0, 0,
#             0.990124, 0.00742879, -0.00960462, 0, 0,
#             0.990187, 0.0094608, -0.010172, 0, 0,
#             0.99372, 0.00560406, -0.00943169, 0, 0,
# Photons
#xcorr
#             1.00506, 1.00697, 1.00595, 1.00595, 1.00595, 1.00595, 1.00595, 
#             0.966651, 0.97381, 0.976516, 0.983254, 0.98502, 0.98502, 0.978472, 
#par
#             0.00132382, 2.17664, -0.00467206, 0.988994, 17.5858,
#             -0.00590257, 1.90733, 0.000684327, 0.986431, 16.6698,
#             0.00265109, 1.73272, -0.00107022, 0.989322, 15.4911,
#             0.00231631, 1.3463, -0.00369555, 0.987133, 10.9233,
#             0.00984253, 1.33889, -0.00392593, 0.979191, 9.35276,
#             0.023683, 1.31198, -0.00947317, 0.963352, 7.5597,
#             0.0851133, 1.38097, -0.0340201, 0.969502, 4.17983,
#             6.71705, 5034.26, -2.68669, 0.970174, 1.00288,
#             1306.82, 472004, -1.86145, 0.981714, -0.25644,
#             0.317121, 3.22717, -0.126848, 0.957792, 2.01028,
#             0.275225, 2.20686, -0.11009, 0.93922, 2.69958,
#             0.0639875, 1.40045, -0.0255853, 0.821566, 7.3297,
#             0.030488, 1.37842, -0.0121879, 0.8173, 9.29944,
#             0.213906, 1.67471, -0.0860589, 0.893636, 3.78218,
## fEt : 7 x 4 (photon/electron, EB/EE)
# Electrons EB
#             0.97213, 0.999528, 5.61192e-06, 0.0143269, -17.1776, 0, 0,
# Electrons EE
#             0.930081,  0.996683,  3.54079e-05,  0.0460187,  -23.2461, 0, 0,
# Photons EB
#             1,  1.00348,  1.001, -9.17302e-06, 0.999688, 0, 0,
# Photons EE
#             1,  0.996931, 0.999497, 0.992617, 7.52128e-05, -1.2845e-07, 1.00231,
## fEnergy : 5 x 2 (photon/electron, EE only)
# Electrons EE
#             400, 0.982475, 4.95413e-05, 0.16886, -30.1517, 
# Photons EE
#             850,  0.994169, 1.28629e-05, 0, 0),
#   2012 May 16
# fEta : p0, p1
             40.2198, -3.03103e-6,
# Electron f(Brem,eta):
             1.00355, 1.00377, 1.00307, 1.00235, 1.0008, 0.999123, 0.995808,
             0.974023, 0.983046, 0.986587, 0.989959, 0.992291, 0.994088, 0.994841,
             
             1.00824, -0.00195358, 0, 0, 0, 
             1.00848, -0.00249326, 0, 0, 0, 
             1.00762, -0.00267961, 0, 0, 0, 
             1.00817, -0.00405541, 0, 0, 0, 
             1.00648, -0.00472328, 0, 0, 0, 
             1.00823, -0.00789251, 0, 0, 0, 
             1.00671, -0.00889114, 0, 0, 0, 
             0.977122, -0.00079133, -0.00213429, 0, 0, 
             0.988986, -0.00383962, -0.00256931, 0, 0, 
             0.990514, 0.00110704, -0.00538053, 0, 0, 
             0.989242,  0.00822155,  -0.00760498,  0,  0, 
             0.99109,  0.0100383,  -0.00889766,  0,  0, 
             0.984981,  0.0207496,  -0.011706,  0,  0, 
             0.996159,  0.00762923,  -0.00876377,  0,  0, 
# Photon f(Brem,eta):
             1.00942, 1.01462, 1.00984, 1.00984, 1.00984, 1.00984, 1.00984,
             0.976343, 0.984129, 0.985861, 0.987185, 0.986922, 0.984653, 0.984653,
                                             
             0.0631272,  2.07465,  -0.0006589,  0.989607,  12.9334, 
             -0.00810258,  1.87803,  0.00312754,  0.989272,  12.777, 
             -0.000777875,  1.6271,  0.0409175,  0.992587,  10.4214, 
             0.00499402,  1.27952,  -0.0171224,  0.990867,  7.31709, 
             0.0206879,  1.3566,  -0.00869229,  0.983379,  7.12404, 
             0.117245,  1.67142,  -0.0468981,  0.986991,  2.89181, 
             0.0855469,  1.42217,  -0.0342187,  0.971139,  4.21491, 
             2.32816,  556.179,  -0.93126,  0.972245,  1.83274, 
             0.462982,  4.21266,  -0.0638084,  0.973512,  1.96724, 
             0.267879,  2.82353,  -0.107158,  0.955956,  2.67778, 
             0.2808,  3.11316,  -0.11232,  0.956383,  2.8149, 
             0.012426,  1.80645,  -1.10844,  0.907241,  4.27577, 
             0.266712,  2.74984,  -0.106685,  0.958985,  2.72102, 
             0.253367,  2.53726,  -0.101347,  0.925249,  3.76083,
## f(ET) Electron EB
             0.976603, 0.999277, 6.91141e-06, 0.0493142, -8.21903, 0, 0,
# f(ET) Electron EE
             0.949713, 1.00196, 3.84843e-06, 0.0329028, -34.6927, 0, 0,
# f(ET) Photon EB
             1, 1.00213, 1.00069, -5.27777e-06, 0.99992, 0, 0,
# f(ET) Photon EE
             1,  1.00206,  0.998431, 0.995999, 3.22962e-05, -1.8556e-08, 1.00205, 
# f(E) Electron EE
             400, 0.986762, 3.65522e-05, 0.178521, -24.8851, 
# f(E) Photon EE
             600,  0.995234,  1.14198e-05,  0, 0),

    producedEcalClusterEnergyCorrectionObjectSpecificParameters = cms.untracked.bool(True),

    producedEcalSampleMask = cms.untracked.bool(True),
    sampleMaskEB = cms.untracked.uint32(1023),
    sampleMaskEE = cms.untracked.uint32(1023)
)
