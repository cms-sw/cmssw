#
# caloStage2Params_2017_v1_10
# change w.r.t. v1_8_4: 92X Layer 1 SF
#
import FWCore.ParameterSet.Config as cms

from L1Trigger.L1TCalorimeter.caloParams_cfi import caloParamsSource
import L1Trigger.L1TCalorimeter.caloParams_cfi
caloStage2Params = L1Trigger.L1TCalorimeter.caloParams_cfi.caloParams.clone()

# towers
caloStage2Params.towerLsbH        = cms.double(0.5)
caloStage2Params.towerLsbE        = cms.double(0.5)
caloStage2Params.towerLsbSum      = cms.double(0.5)
caloStage2Params.towerNBitsH      = cms.int32(8)
caloStage2Params.towerNBitsE      = cms.int32(8)
caloStage2Params.towerNBitsSum    = cms.int32(9)
caloStage2Params.towerNBitsRatio  = cms.int32(3)
caloStage2Params.towerEncoding    = cms.bool(True)

# regions
caloStage2Params.regionLsb        = cms.double(0.5)
caloStage2Params.regionPUSType    = cms.string("None")
caloStage2Params.regionPUSParams  = cms.vdouble()

# EG
caloStage2Params.egLsb                      = cms.double(0.5)
caloStage2Params.egSeedThreshold            = cms.double(2.)
caloStage2Params.egNeighbourThreshold       = cms.double(1.)
caloStage2Params.egHcalThreshold            = cms.double(0.)
caloStage2Params.egTrimmingLUTFile          = cms.FileInPath("L1Trigger/L1TCalorimeter/data/egTrimmingLUT_10_v16.01.19.txt")
caloStage2Params.egMaxHcalEt                = cms.double(0.)
caloStage2Params.egMaxPtHOverE          = cms.double(128.)
caloStage2Params.egMaxHOverELUTFile         = cms.FileInPath("L1Trigger/L1TCalorimeter/data/HoverEIdentification_0.995_v15.12.23.txt")
caloStage2Params.egBypassExtHOverE          = cms.uint32(0)
caloStage2Params.egCompressShapesLUTFile    = cms.FileInPath("L1Trigger/L1TCalorimeter/data/egCompressLUT_v4.txt")
caloStage2Params.egShapeIdType              = cms.string("compressed")
caloStage2Params.egShapeIdVersion           = cms.uint32(0)
caloStage2Params.egShapeIdLUTFile           = cms.FileInPath("L1Trigger/L1TCalorimeter/data/shapeIdentification_adapt0.99_compressedieta_compressedE_compressedshape_v15.12.08.txt")#Not used any more in the current emulator version, merged with calibration LUT

caloStage2Params.egPUSType                  = cms.string("None")
caloStage2Params.egIsolationType            = cms.string("compressed")
caloStage2Params.egIsoLUTFile               = cms.FileInPath("L1Trigger/L1TCalorimeter/data/EG_Iso_LUT_04_04_2017.txt")
caloStage2Params.egIsoLUTFile2               = cms.FileInPath("L1Trigger/L1TCalorimeter/data/EG_LooseIsoIdentification_adapt_extrap_FW_v16.08.08.2.txt")
caloStage2Params.egIsoAreaNrTowersEta       = cms.uint32(2)
caloStage2Params.egIsoAreaNrTowersPhi       = cms.uint32(4)
caloStage2Params.egIsoVetoNrTowersPhi       = cms.uint32(2)
#caloStage2Params.egIsoPUEstTowerGranularity = cms.uint32(1)
#caloStage2Params.egIsoMaxEtaAbsForTowerSum  = cms.uint32(4)
#caloStage2Params.egIsoMaxEtaAbsForIsoSum    = cms.uint32(27)
caloStage2Params.egPUSParams                = cms.vdouble(1,4,32) #Isolation window in firmware goes up to abs(ieta)=32 for now
caloStage2Params.egCalibrationType          = cms.string("compressed")
caloStage2Params.egCalibrationVersion       = cms.uint32(0)
caloStage2Params.egCalibrationLUTFile       = cms.FileInPath("L1Trigger/L1TCalorimeter/data/EG_Calibration_LUT_FW_v17.04.04_shapeIdentification_adapt0.99_compressedieta_compressedE_compressedshape_v15.12.08_correct.txt")


# Tau
caloStage2Params.tauLsb                        = cms.double(0.5)
caloStage2Params.tauSeedThreshold              = cms.double(0.)
caloStage2Params.tauNeighbourThreshold         = cms.double(0.)
caloStage2Params.tauIsoAreaNrTowersEta         = cms.uint32(2)
caloStage2Params.tauIsoAreaNrTowersPhi         = cms.uint32(4)
caloStage2Params.tauIsoVetoNrTowersPhi         = cms.uint32(2)
caloStage2Params.tauPUSType                    = cms.string("None")
caloStage2Params.tauIsoLUTFile                 = cms.FileInPath("L1Trigger/L1TCalorimeter/data/Tau_Iso_LUT_Option_22_2017_FW_v10_261017.0.0.txt")
caloStage2Params.tauIsoLUTFile2                = cms.FileInPath("L1Trigger/L1TCalorimeter/data/Tau_Iso_LUT_Option_22_2017_FW_v10_261017.0.0.txt")
caloStage2Params.tauCalibrationLUTFile         = cms.FileInPath("L1Trigger/L1TCalorimeter/data/Tau_Calibration_LUT_261017.0.0.txt")
caloStage2Params.tauCompressLUTFile            = cms.FileInPath("L1Trigger/L1TCalorimeter/data/tauCompressAllLUT_12bit_v3.txt")
caloStage2Params.tauPUSParams                  = cms.vdouble(1,4,32)

# jets
caloStage2Params.jetLsb                = cms.double(0.5)
caloStage2Params.jetSeedThreshold      = cms.double(4.0)
caloStage2Params.jetNeighbourThreshold = cms.double(0.)
caloStage2Params.jetPUSType            = cms.string("ChunkyDonut")
caloStage2Params.jetBypassPUS          = cms.uint32(0)

# Calibration options
# function6PtParams22EtaBins or None
#caloStage2Params.jetCalibrationType    = cms.string("None")
#caloStage2Params.jetCalibrationType = cms.string("function8PtParams22EtaBins")
caloStage2Params.jetCalibrationType = cms.string("LUT")

#Vector with 6 parameters for eta bin, from low eta to high
# 1,0,1,0,1,1 gives no correction
# must be in this form as may require > 255 arguments

# Or vector with 8 parameters, which caps correction value below given pT
# as 6 parameters, but last two are max correction and L1 pT below which cap is applied, respectively

jetCalibParamsVector = cms.vdouble()
jetCalibParamsVector.extend([
        1,0,1,0,1,1,1.36123039014,1024,
        1,0,1,0,1,1,1.37830172245,1024,
        1,0,1,0,1,1,1.37157036457,1024,
        1,0,1,0,1,1,1.42460009989,1024,
        10.1179757811,-697.422255848,55.9767511168,599.040770412,0.00930772659892,-21.9921521313,1.77585386314,24.1202894336,
        12.2578170485,-736.96846599,45.3225355911,848.976802835,0.00946235693865,-21.7970133915,2.04623980351,19.6049149791,
        14.0198255047,-769.175319944,38.687351315,1072.9785137,0.00951954709279,-21.6277409602,2.08021511285,22.265051562,
        14.119589176,-766.199501821,38.7767169666,1059.63374337,0.00952979125289,-21.6477483043,2.05901166216,23.8125466978,
        13.7594864391,-761.860391454,39.9060363401,1019.30588542,0.00952105483129,-21.6814176696,2.03808638982,22.2127275989,
        10.2635352836,-466.890522023,32.5408463829,2429.03382746,0.0111274121697,-22.0890253377,2.04880080215,22.5083699943,
        5.46086027683,-150.888778124,18.3292242153,16968.6469599,0.0147496053457,-22.4089831889,2.08107691501,22.4129703515,
        5.46086027683,-150.888778124,18.3292242153,16968.6469599,0.0147496053457,-22.4089831889,2.08107691501,22.4129703515,
        10.2635352836,-466.890522023,32.5408463829,2429.03382746,0.0111274121697,-22.0890253377,2.04880080215,22.5083699943,
        13.7594864391,-761.860391454,39.9060363401,1019.30588542,0.00952105483129,-21.6814176696,2.03808638982,22.2127275989,
        14.119589176,-766.199501821,38.7767169666,1059.63374337,0.00952979125289,-21.6477483043,2.05901166216,23.8125466978,
        14.0198255047,-769.175319944,38.687351315,1072.9785137,0.00951954709279,-21.6277409602,2.08021511285,22.265051562,
        12.2578170485,-736.96846599,45.3225355911,848.976802835,0.00946235693865,-21.7970133915,2.04623980351,19.6049149791,
        10.1179757811,-697.422255848,55.9767511168,599.040770412,0.00930772659892,-21.9921521313,1.77585386314,24.1202894336,
        1,0,1,0,1,1,1.42460009989,1024,
        1,0,1,0,1,1,1.37157036457,1024,
        1,0,1,0,1,1,1.37830172245,1024,
        1,0,1,0,1,1,1.36123039014,1024
])
caloStage2Params.jetCalibrationParams  = jetCalibParamsVector 

caloStage2Params.jetCompressPtLUTFile     = cms.FileInPath("L1Trigger/L1TCalorimeter/data/lut_pt_compress_2017v1.txt")
caloStage2Params.jetCompressEtaLUTFile    = cms.FileInPath("L1Trigger/L1TCalorimeter/data/lut_eta_compress_2017v1.txt")
caloStage2Params.jetCalibrationLUTFile    = cms.FileInPath("L1Trigger/L1TCalorimeter/data/lut_calib_2017v3_mean.txt")


# sums: 0=ET, 1=HT, 2=MET, 3=MHT
caloStage2Params.etSumLsb                = cms.double(0.5)
caloStage2Params.etSumEtaMin             = cms.vint32(1, 1, 1, 1, 1)
caloStage2Params.etSumEtaMax             = cms.vint32(28,  26, 28,  26, 28)
caloStage2Params.etSumEtThreshold        = cms.vdouble(0.,  30.,  0.,  30., 0.) # only 2nd (HT) and 4th (MHT) values applied
caloStage2Params.etSumMetPUSType         = cms.string("LUT") # et threshold from this LUT supercedes et threshold in line above
caloStage2Params.etSumEttPUSType         = cms.string("None")
caloStage2Params.etSumEcalSumPUSType     = cms.string("None")
caloStage2Params.etSumBypassMetPUS       = cms.uint32(0)
caloStage2Params.etSumBypassEttPUS       = cms.uint32(1)
caloStage2Params.etSumBypassEcalSumPUS    = cms.uint32(1)
caloStage2Params.etSumXCalibrationType    = cms.string("None")
caloStage2Params.etSumYCalibrationType    = cms.string("None")
caloStage2Params.etSumEttCalibrationType  = cms.string("None")
caloStage2Params.etSumEcalSumCalibrationType = cms.string("None")

caloStage2Params.etSumMetPUSLUTFile               = cms.FileInPath("L1Trigger/L1TCalorimeter/data/lut_towEtThresh_2017v7.txt")
caloStage2Params.etSumEttPUSLUTFile               = cms.FileInPath("L1Trigger/L1TCalorimeter/data/lut_towEtThresh_dummy.txt")
caloStage2Params.etSumEcalSumPUSLUTFile           = cms.FileInPath("L1Trigger/L1TCalorimeter/data/lut_towEtThresh_dummy.txt")
caloStage2Params.etSumXCalibrationLUTFile         = cms.FileInPath("L1Trigger/L1TCalorimeter/data/lut_etSumPUS_dummy.txt")
caloStage2Params.etSumYCalibrationLUTFile         = cms.FileInPath("L1Trigger/L1TCalorimeter/data/lut_etSumPUS_dummy.txt")
caloStage2Params.etSumEttCalibrationLUTFile       = cms.FileInPath("L1Trigger/L1TCalorimeter/data/lut_etSumPUS_dummy.txt")
caloStage2Params.etSumEcalSumCalibrationLUTFile   = cms.FileInPath("L1Trigger/L1TCalorimeter/data/lut_etSumPUS_dummy.txt")

# Layer 1 LUT specification
#
# Et-dependent scale factors
# ECal/HCal scale factors will be a 13*28 array:
#   28 eta scale factors (1-28)
#   in 13 ET bins (6, 9, 12, 15, 20, 25, 30, 35, 40, 45, 55, 70, Max)
#  So, index = etBin*28+ieta
# ECAL and HCAl calibrations using mean
caloStage2Params.layer1ECalScaleETBins = cms.vint32([6, 9, 12, 15, 20, 25, 30, 35, 40, 45, 55, 70, 256])
caloStage2Params.layer1ECalScaleFactors = cms.vdouble([
        1.197340, 1.198549, 1.168785, 1.173931, 1.191020, 1.209413, 1.196497, 1.209573, 1.195505, 1.231375, 1.235413, 1.244471, 1.283982, 1.325228, 1.334809, 1.353722, 1.428926, 2.126767, 
        1.450591, 1.589677, 1.580657, 1.629203, 1.564859, 1.577755, 1.625670, 1.594695, 1.424415, 1.321468, 1.135290, 1.151154, 1.125139, 1.130923, 1.135517, 1.148669, 1.147089, 1.154148, 
        1.183942, 1.187542, 1.191086, 1.190894, 1.249920, 1.258438, 1.273714, 1.287786, 1.342814, 2.053505, 1.313293, 1.461993, 1.451037, 1.465911, 1.438294, 1.455272, 1.501573, 1.477581, 
        1.339441, 1.245791, 1.129958, 1.107732, 1.102933, 1.100946, 1.120345, 1.124828, 1.126518, 1.136332, 1.145752, 1.175010, 1.179295, 1.188173, 1.211749, 1.224195, 1.234790, 1.239917, 
        1.353503, 2.008072, 1.252317, 1.365824, 1.378117, 1.403996, 1.356526, 1.385768, 1.434346, 1.415377, 1.298908, 1.216760, 1.102309, 1.097450, 1.090676, 1.084893, 1.091920, 1.109602, 
        1.103849, 1.112758, 1.126005, 1.137318, 1.120697, 1.142343, 1.150537, 1.201907, 1.168302, 1.188819, 1.228637, 1.936608, 1.224452, 1.326251, 1.342814, 1.353976, 1.325363, 1.359490, 
        1.399696, 1.364164, 1.276219, 1.195622, 1.076251, 1.069282, 1.066564, 1.074088, 1.070074, 1.084258, 1.086150, 1.076595, 1.092879, 1.114732, 1.101672, 1.105921, 1.119918, 1.145530, 
        1.167513, 1.147558, 1.191129, 1.809826, 1.202365, 1.287467, 1.304235, 1.317980, 1.291666, 1.317809, 1.374505, 1.342310, 1.254258, 1.175981, 1.061569, 1.053739, 1.050862, 1.052114, 
        1.057964, 1.073229, 1.058238, 1.066881, 1.063274, 1.090312, 1.075247, 1.088771, 1.097769, 1.135655, 1.119135, 1.123404, 1.172366, 1.741823, 1.173261, 1.258103, 1.279940, 1.279914, 
        1.276035, 1.291460, 1.347826, 1.321888, 1.237275, 1.159756, 1.058557, 1.043179, 1.038852, 1.040351, 1.047275, 1.056788, 1.051126, 1.058392, 1.051716, 1.085330, 1.061614, 1.073405, 
        1.081882, 1.109701, 1.103221, 1.100014, 1.149658, 1.650972, 1.163525, 1.237588, 1.259934, 1.268718, 1.254323, 1.276469, 1.335477, 1.298039, 1.226921, 1.151347, 1.046273, 1.035069, 
        1.033646, 1.034902, 1.037039, 1.055578, 1.043272, 1.044873, 1.045536, 1.067714, 1.058866, 1.060444, 1.067633, 1.101122, 1.083575, 1.089725, 1.133219, 1.530750, 1.150335, 1.220118, 
        1.237836, 1.251671, 1.239206, 1.262410, 1.317311, 1.279968, 1.221607, 1.145441, 1.039182, 1.033807, 1.026964, 1.030851, 1.035037, 1.046218, 1.034010, 1.038878, 1.038807, 1.061946, 
        1.047964, 1.052194, 1.061816, 1.089591, 1.077566, 1.075823, 1.118349, 1.441061, 1.144726, 1.205469, 1.228561, 1.240078, 1.224216, 1.249805, 1.307356, 1.275350, 1.210373, 1.139566, 
        1.033242, 1.027776, 1.025388, 1.025144, 1.029551, 1.045796, 1.031684, 1.032839, 1.032635, 1.060448, 1.040870, 1.047611, 1.060231, 1.075297, 1.066971, 1.073752, 1.113008, 1.383509, 
        1.129704, 1.198243, 1.222456, 1.234389, 1.224164, 1.243444, 1.294541, 1.265006, 1.178805, 1.135663, 1.029008, 1.023628, 1.019729, 1.022226, 1.024997, 1.036473, 1.027582, 1.028378, 
        1.029302, 1.047454, 1.035725, 1.038674, 1.047384, 1.068694, 1.060923, 1.063771, 1.100034, 1.333569, 1.126848, 1.185826, 1.209725, 1.224937, 1.212785, 1.236321, 1.284212, 1.256900, 
        1.115347, 1.114443, 1.023628, 1.017810, 1.014326, 1.015847, 1.018518, 1.028086, 1.020245, 1.020984, 1.022730, 1.038105, 1.027760, 1.028804, 1.041350, 1.059088, 1.051748, 1.053073, 
        1.087165, 1.252114, 1.119432, 1.174365, 1.196021, 1.210201, 1.200302, 1.226177, 1.270829, 1.244451, 1.048434, 1.049180, 1.018333, 1.014078, 1.010072, 1.010963, 1.013350, 1.020835, 
        1.014829, 1.016063, 1.016330, 1.026939, 1.021395, 1.022569, 1.033490, 1.047872, 1.042920, 1.044526, 1.072217, 1.185529, 1.108676, 1.161552, 1.183706, 1.197698, 1.189131, 1.212932, 
        1.255325, 1.225494, 1.048434, 1.049180
        
    ])
caloStage2Params.layer1HCalScaleETBins = cms.vint32([6, 9, 12, 15, 20, 25, 30, 35, 40, 45, 55, 70, 256])
caloStage2Params.layer1HCalScaleFactors = cms.vdouble([
        1.488772, 1.486679, 1.482133, 1.479425, 1.485548, 1.493674, 1.492273, 1.493985, 1.492969, 1.509587, 1.506320, 1.515023, 1.536133, 1.531514, 1.526063, 1.523588, 1.484326, 1.331186, 
        1.355782, 1.387601, 1.342361, 1.360238, 1.360894, 1.357810, 1.361534, 1.375109, 1.424183, 1.501489, 1.399359, 1.404418, 1.398637, 1.399352, 1.396019, 1.410175, 1.410339, 1.406340, 
        1.407406, 1.418949, 1.419240, 1.429573, 1.439777, 1.439575, 1.437873, 1.429671, 1.386724, 1.231026, 1.273743, 1.302278, 1.247894, 1.253293, 1.255920, 1.251581, 1.251463, 1.265636, 
        1.304193, 1.359426, 1.344773, 1.350364, 1.344524, 1.345861, 1.341056, 1.353025, 1.354453, 1.351831, 1.347695, 1.364280, 1.359560, 1.372041, 1.381087, 1.385518, 1.382776, 1.370359, 
        1.327976, 1.177840, 1.228646, 1.249099, 1.186989, 1.193231, 1.197696, 1.195938, 1.196179, 1.206994, 1.244052, 1.290444, 1.312420, 1.314244, 1.309209, 1.307359, 1.307022, 1.316532, 
        1.318803, 1.313482, 1.308246, 1.323321, 1.325338, 1.330967, 1.337016, 1.338398, 1.339131, 1.327637, 1.286923, 1.141686, 1.190420, 1.213207, 1.149381, 1.160818, 1.159674, 1.159706, 
        1.158536, 1.169460, 1.207328, 1.248669, 1.276808, 1.278511, 1.274205, 1.271484, 1.270841, 1.278961, 1.282849, 1.277440, 1.273669, 1.284206, 1.284441, 1.290392, 1.294976, 1.296487, 
        1.298681, 1.286720, 1.244613, 1.110049, 1.157259, 1.176192, 1.112071, 1.119705, 1.123068, 1.121734, 1.123006, 1.132017, 1.169278, 1.213867, 1.242737, 1.243424, 1.240171, 1.239669, 
        1.236894, 1.241291, 1.244473, 1.241839, 1.234634, 1.244791, 1.243586, 1.250908, 1.250071, 1.254379, 1.257426, 1.244129, 1.200212, 1.077383, 1.122736, 1.139789, 1.076388, 1.083750, 
        1.085063, 1.085238, 1.086152, 1.095831, 1.131103, 1.174074, 1.215358, 1.216519, 1.212013, 1.211151, 1.210772, 1.213001, 1.216205, 1.212945, 1.203300, 1.212112, 1.212353, 1.216219, 
        1.216911, 1.220303, 1.222827, 1.209306, 1.164908, 1.053285, 1.098127, 1.112139, 1.046242, 1.053812, 1.054951, 1.055403, 1.056634, 1.065248, 1.100811, 1.146619, 1.189579, 1.190152, 
        1.186635, 1.187759, 1.184085, 1.184657, 1.188523, 1.186424, 1.177457, 1.183637, 1.182490, 1.187512, 1.187172, 1.190456, 1.192421, 1.180374, 1.138839, 1.034745, 1.078450, 1.089012, 
        1.021600, 1.028598, 1.029529, 1.030437, 1.033001, 1.039217, 1.075602, 1.118267, 1.171107, 1.168946, 1.166512, 1.166769, 1.161480, 1.165436, 1.165121, 1.162166, 1.153355, 1.158267, 
        1.159683, 1.162556, 1.161758, 1.164033, 1.169004, 1.154110, 1.114707, 1.016696, 1.060155, 1.070569, 1.000000, 1.005364, 1.007959, 1.009434, 1.009694, 1.015478, 1.051155, 1.095691, 
        1.147927, 1.150166, 1.146134, 1.147374, 1.142142, 1.143955, 1.144191, 1.141270, 1.134016, 1.138813, 1.136992, 1.142244, 1.139741, 1.140879, 1.146482, 1.132095, 1.091087, 1.003826, 
        1.042366, 1.053090, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.023234, 1.031095, 1.075333, 1.124485, 1.126611, 1.122901, 1.121996, 1.119331, 1.121150, 1.122024, 1.116685, 
        1.110000, 1.112285, 1.113655, 1.114063, 1.112371, 1.111978, 1.116022, 1.101930, 1.061707, 1.000000, 1.024583, 1.031882, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.055449, 
        1.003816, 1.046213, 1.092452, 1.092536, 1.091150, 1.090404, 1.085964, 1.085791, 1.086160, 1.082304, 1.075379, 1.074526, 1.072966, 1.073412, 1.070047, 1.069312, 1.070556, 1.054325, 
        1.019816, 1.000000, 1.000000, 1.001951, 1.000000, 1.000000, 1.000000, 1.000000, 1.000301, 1.032098, 1.000000, 1.005659, 1.051117, 1.050717, 1.049425, 1.047891, 1.044951, 1.044487, 
        1.042311, 1.036290, 1.030471, 1.028289, 1.022935, 1.020965, 1.017667, 1.013806, 1.014022, 1.004382, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 
        1.000000, 1.000000, 1.000000, 1.000000
    ])
# HF 1x1 scale factors will be a 13*12 array:
#  12 eta scale factors (30-41)
#  in 13 ET bins (6, 9, 12, 15, 20, 25, 30, 35, 40, 45, 55, 70, Max)
#  So, index = etBin*12+ietaHF
# HF energies were formerly multiplied by 0.7; this has been removed
caloStage2Params.layer1HFScaleETBins = cms.vint32([6, 9, 12, 15, 20, 25, 30, 35, 40, 45, 55, 70, 256])
caloStage2Params.layer1HFScaleFactors = cms.vdouble([
    2.378339, 1.502094, 1.558828, 1.468909, 1.388092, 1.444754, 1.493556, 1.541491, 1.647650, 1.812072, 2.791145, 2.844066,
    2.111653, 1.312496, 1.351124, 1.291042, 1.239054, 1.278956, 1.315620, 1.361558, 1.449292, 1.571425, 2.709180, 2.717564,
    1.963179, 1.217324, 1.256356, 1.202818, 1.162660, 1.204208, 1.231526, 1.276481, 1.351362, 1.457253, 2.613049, 2.644112,
    1.864273, 1.162345, 1.199680, 1.153738, 1.119396, 1.152063, 1.182551, 1.225995, 1.291988, 1.390649, 2.529912, 2.581591,
    1.752451, 1.117623, 1.147027, 1.110546, 1.079779, 1.114737, 1.142444, 1.178901, 1.242175, 1.336171, 2.407025, 2.526142,
    1.663160, 1.074331, 1.106646, 1.072905, 1.049034, 1.080200, 1.108287, 1.143216, 1.199594, 1.291001, 2.232567, 2.450402,
    1.573166, 1.048392, 1.078650, 1.048091, 1.024573, 1.055920, 1.081953, 1.115248, 1.170655, 1.256432, 2.070575, 2.389922,
    1.489765, 1.024323, 1.055465, 1.029036, 1.007379, 1.036369, 1.061089, 1.092431, 1.145947, 1.227190, 1.925361, 2.348549,
    1.404872, 1.006701, 1.035613, 1.009332, 1.007379, 1.017418, 1.040979, 1.071060, 1.120826, 1.197973, 1.791211, 2.243741,
    1.339055, 1.006701, 1.019214, 1.009332, 1.007379, 1.003242, 1.026977, 1.054007, 1.099699, 1.168445, 1.688074, 2.103020,
    1.272889, 1.006701, 1.006044, 1.009332, 1.007379, 1.003242, 1.009030, 1.033555, 1.074019, 1.135660, 1.573541, 1.918549,
    1.188140, 1.006701, 1.006044, 1.009332, 1.007379, 1.003242, 1.009030, 1.001923, 1.039081, 1.094883, 1.434509, 1.705331,
    1.108268, 1.006701, 1.006044, 1.009332, 1.007379, 1.003242, 1.009030, 1.001923, 1.010006, 1.057960, 1.301315, 1.523940
    ])
    
