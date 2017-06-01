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
#caloStage2Params.egIsoLUTFile               = cms.FileInPath("L1Trigger/L1TCalorimeter/data/IsoIdentification_adapt_extrap_v16.07.29.txt")
caloStage2Params.egIsoLUTFile               = cms.FileInPath("L1Trigger/L1TCalorimeter/data/EG_Iso_LUT_04_04_2017.txt")
#caloStage2Params.egIsoLUTFile2               = cms.FileInPath("L1Trigger/L1TCalorimeter/data/IsoIdentification_adapt_extrap_FW_v16.08.08.txt")
caloStage2Params.egIsoLUTFile2              = cms.FileInPath("L1Trigger/L1TCalorimeter/data/EG_Iso_LUT_04_04_2017.txt")
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
caloStage2Params.tauIsoLUTFile                 = cms.FileInPath("L1Trigger/L1TCalorimeter/data/Tau_Iso_LUT_Option_22_2017_FW_v9.0.0.txt")
caloStage2Params.tauIsoLUTFile2                = cms.FileInPath("L1Trigger/L1TCalorimeter/data/Tau_Iso_LUT_Option_22_2017_FW_v9.0.0.txt")
caloStage2Params.tauCalibrationLUTFile         = cms.FileInPath("L1Trigger/L1TCalorimeter/data/Tau_Calibration_LUT_2017_Layer1Calibration_FW_v12.0.0.txt")
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
caloStage2Params.jetCalibrationLUTFile    = cms.FileInPath("L1Trigger/L1TCalorimeter/data/lut_calib_2017v1.txt")


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

caloStage2Params.etSumMetPUSLUTFile               = cms.FileInPath("L1Trigger/L1TCalorimeter/data/lut_towEtThresh_2017v4.txt")
caloStage2Params.etSumEttPUSLUTFile               = cms.FileInPath("L1Trigger/L1TCalorimeter/data/lut_towEtThresh_dummy.txt")
caloStage2Params.etSumEcalSumPUSLUTFile           = cms.FileInPath("L1Trigger/L1TCalorimeter/data/lut_towEtThresh_dummy.txt")
caloStage2Params.etSumXCalibrationLUTFile         = cms.FileInPath("L1Trigger/L1TCalorimeter/data/lut_etSumPUS_dummy.txt")
caloStage2Params.etSumYCalibrationLUTFile         = cms.FileInPath("L1Trigger/L1TCalorimeter/data/lut_etSumPUS_dummy.txt")
caloStage2Params.etSumEttCalibrationLUTFile       = cms.FileInPath("L1Trigger/L1TCalorimeter/data/lut_etSumPUS_dummy.txt")
caloStage2Params.etSumEcalSumCalibrationLUTFile   = cms.FileInPath("L1Trigger/L1TCalorimeter/data/lut_etSumPUS_dummy.txt")

# Layer 1 LUT specification
#
# Et-dependent scale factors
# ECal/HCal scale factors will be a 9*28 array:
#   28 eta scale factors (1-28)
#   in 9 ET bins (10, 15, 20, 25, 30, 35, 40, 45, Max)
#  So, index = etBin*28+ieta
#FInal ecal and HCAl calibrations using mean.. 
#caloStage2Params.layer1ECalScaleETBins = cms.vint32([10, 15, 20, 25, 30, 35, 40, 45, 256])
#92X ecal and hcal calibr
caloStage2Params.layer1ECalScaleETBins = cms.vint32([6, 9, 12, 15, 20, 25, 30, 35, 40, 45, 55, 70, 256])
caloStage2Params.layer1ECalScaleFactors = cms.vdouble([
        1.197952, 1.199612, 1.171811, 1.188309, 1.185184, 1.207810, 1.182339, 1.193361, 1.202947, 1.242417, 1.217694, 1.239285, 1.261151, 1.343206, 1.323074, 1.371586, 
        1.426672, 2.144334, 1.429348, 1.594896, 1.600363, 1.604834, 1.558857, 1.578504, 1.638347, 1.607156, 1.431742, 1.327736, 1.140528, 1.138384, 1.121035, 1.127105, 1.135175, 1.136278, 
        1.130008, 1.143710, 1.141468, 1.166651, 1.155257, 1.168006, 1.198731, 1.233869, 1.233772, 1.225326, 1.339436, 2.033770, 1.320422, 1.443975, 1.456446, 1.477599, 1.441075, 1.457994, 
        1.511734, 1.476617, 1.340335, 1.249196, 1.126286, 1.104403, 1.093190, 1.090319, 1.095618, 1.107452, 1.103129, 1.100107, 1.107142, 1.124129, 1.125729, 1.124254, 1.145309, 1.175804, 
        1.179267, 1.186265, 1.285031, 1.924260, 1.261563, 1.371971, 1.382497, 1.398012, 1.365497, 1.390656, 1.436558, 1.417432, 1.302307, 1.220363, 1.098332, 1.080232, 1.075614, 1.076260, 
        1.087070, 1.084200, 1.083463, 1.084395, 1.089545, 1.121732, 1.095221, 1.103458, 1.120284, 1.148391, 1.148974, 1.146549, 1.214708, 1.877223, 1.224579, 1.313952, 1.338085, 1.343860, 
        1.323828, 1.349932, 1.405292, 1.372595, 1.278908, 1.196521, 1.075825, 1.062361, 1.062015, 1.063174, 1.069308, 1.081125, 1.072951, 1.069653, 1.073877, 1.092260, 1.085669, 1.083712, 
        1.099936, 1.128952, 1.117871, 1.126101, 1.178040, 1.791865, 1.199733, 1.292565, 1.298237, 1.322715, 1.295345, 1.319737, 1.371282, 1.337179, 1.262159, 1.177947, 1.062509, 1.054907, 
        1.049096, 1.047346, 1.053254, 1.066058, 1.056319, 1.057785, 1.059969, 1.081881, 1.064681, 1.069310, 1.081193, 1.104527, 1.100641, 1.102750, 1.153384, 1.800766, 1.170628, 1.244467, 
        1.267245, 1.289284, 1.270600, 1.293784, 1.353725, 1.311643, 1.241849, 1.163381, 1.054006, 1.041832, 1.038100, 1.041862, 1.043342, 1.057659, 1.045793, 1.047931, 1.052328, 1.066124, 
        1.053602, 1.059841, 1.069307, 1.095555, 1.083182, 1.090375, 1.128733, 1.606569, 1.162686, 1.234100, 1.251488, 1.261390, 1.251928, 1.275649, 1.330015, 1.297625, 1.231105, 1.153933, 
        1.047649, 1.035181, 1.031156, 1.036037, 1.037613, 1.048532, 1.040916, 1.042513, 1.039832, 1.060550, 1.047866, 1.051857, 1.062599, 1.083945, 1.073368, 1.077310, 1.116720, 1.517123, 
        1.146415, 1.221461, 1.240530, 1.251901, 1.239145, 1.262144, 1.317186, 1.285539, 1.221345, 1.147579, 1.038221, 1.033990, 1.030587, 1.028780, 1.031060, 1.042939, 1.032373, 1.036429, 
        1.033829, 1.054052, 1.042072, 1.042584, 1.055777, 1.074383, 1.068770, 1.069990, 1.115191, 1.438450, 1.143701, 1.210090, 1.229611, 1.241863, 1.228984, 1.254195, 1.304402, 1.275210, 
        1.210725, 1.141448, 1.033962, 1.031849, 1.024920, 1.026496, 1.028343, 1.041070, 1.028119, 1.032854, 1.033208, 1.046729, 1.039219, 1.038556, 1.050553, 1.071673, 1.061654, 1.062923, 
        1.105704, 1.385463, 1.133946, 1.200775, 1.219563, 1.233578, 1.221764, 1.246174, 1.297576, 1.269471, 1.180247, 1.136472, 1.029376, 1.024117, 1.020323, 1.020280, 1.023976, 1.035936, 
        1.025610, 1.027397, 1.027854, 1.041937, 1.032590, 1.034393, 1.045698, 1.062831, 1.055954, 1.058898, 1.093065, 1.311626, 1.126779, 1.187125, 1.206374, 1.222170, 1.211284, 1.234981, 
        1.284902, 1.257305, 1.115701, 1.115225, 1.023642, 1.019021, 1.014943, 1.015169, 1.018963, 1.027962, 1.019170, 1.021647, 1.021166, 1.033605, 1.026753, 1.027090, 1.038400, 1.056126, 
        1.048641, 1.049684, 1.083990, 1.256040, 1.117390, 1.175256, 1.197354, 1.207801, 1.202361, 1.224181, 1.268952, 1.248606, 1.049506, 1.047856, 1.018408, 1.013553, 1.009873, 1.010861, 
        1.013761, 1.020587, 1.014378, 1.016465, 1.016320, 1.025128, 1.020897, 1.022061, 1.032362, 1.046759, 1.042262, 1.042801, 1.071554, 1.186657, 1.109567, 1.162378, 1.183345, 1.197062, 
        1.189865, 1.212548, 1.254481, 1.221767, 1.049506, 1.047856,
        
    ])
caloStage2Params.layer1HCalScaleETBins = cms.vint32([6, 9, 12, 15, 20, 25, 30, 35, 40, 45, 55, 70, 256])
caloStage2Params.layer1HCalScaleFactors = cms.vdouble([
        1.499079, 1.494351, 1.495460, 1.489769, 1.496830, 1.508075, 1.506299, 1.503138, 1.508714, 1.517661, 1.520965, 1.529423, 1.548687, 1.535266, 1.539168, 1.536947, 1.489745, 1.347200, 
        1.361355, 1.396385, 1.337087, 1.354658, 1.356307, 1.353401, 1.355467, 1.368961, 1.420251, 1.497840, 1.396092, 1.402071, 1.397805, 1.396997, 1.394375, 1.404777, 1.405527, 1.403687, 
        1.402119, 1.413829, 1.417454, 1.427746, 1.434836, 1.430412, 1.438483, 1.427156, 1.381621, 1.229879, 1.271887, 1.298427, 1.243404, 1.249303, 1.252799, 1.248931, 1.247246, 1.261320, 
        1.301001, 1.357177, 1.340589, 1.347123, 1.339676, 1.344809, 1.339274, 1.348966, 1.350059, 1.348007, 1.343982, 1.360177, 1.356472, 1.367041, 1.374830, 1.375483, 1.380418, 1.367383, 
        1.322256, 1.177480, 1.224817, 1.247885, 1.184237, 1.190086, 1.194819, 1.193867, 1.193100, 1.204249, 1.241620, 1.289157, 1.309500, 1.311204, 1.305928, 1.304929, 1.304077, 1.311633, 
        1.314597, 1.310174, 1.306947, 1.318523, 1.323998, 1.325470, 1.330777, 1.332675, 1.335317, 1.325278, 1.279659, 1.142544, 1.187826, 1.211386, 1.147240, 1.158471, 1.157796, 1.158730, 
        1.156373, 1.167160, 1.205418, 1.246752, 1.273098, 1.275073, 1.270113, 1.268909, 1.268470, 1.277336, 1.277057, 1.273970, 1.269818, 1.281694, 1.282820, 1.286036, 1.289149, 1.290108, 
        1.294726, 1.283059, 1.239843, 1.110232, 1.153937, 1.174723, 1.110603, 1.118901, 1.121686, 1.120737, 1.121537, 1.129848, 1.167677, 1.212655, 1.238363, 1.239264, 1.237167, 1.236010, 
        1.234335, 1.238381, 1.239978, 1.237023, 1.231497, 1.241121, 1.241507, 1.246289, 1.246535, 1.250265, 1.254867, 1.241189, 1.195854, 1.078607, 1.120689, 1.139331, 1.075725, 1.081924, 
        1.084386, 1.085120, 1.085153, 1.094534, 1.129776, 1.172897, 1.211913, 1.213610, 1.210371, 1.207700, 1.205847, 1.209644, 1.210814, 1.208364, 1.201133, 1.209853, 1.209213, 1.212815, 
        1.212367, 1.215195, 1.219575, 1.206021, 1.160050, 1.055492, 1.096818, 1.110448, 1.046707, 1.053339, 1.055143, 1.055903, 1.056550, 1.064473, 1.100824, 1.145933, 1.187377, 1.187777, 
        1.184066, 1.186466, 1.180503, 1.182323, 1.186816, 1.182680, 1.173752, 1.182839, 1.179816, 1.183264, 1.183115, 1.185115, 1.189050, 1.177245, 1.133626, 1.035847, 1.075723, 1.088670, 
        1.022256, 1.028320, 1.030409, 1.031382, 1.033072, 1.038624, 1.075508, 1.118184, 1.167103, 1.165591, 1.163772, 1.161628, 1.159692, 1.163233, 1.161176, 1.158051, 1.151840, 1.155717, 
        1.155762, 1.158094, 1.156992, 1.159611, 1.165919, 1.152787, 1.110360, 1.017816, 1.056909, 1.070223, 1.000048, 1.006319, 1.008273, 1.010241, 1.009674, 1.015772, 1.051577, 1.096107, 
        1.147513, 1.146248, 1.145148, 1.144969, 1.139576, 1.142478, 1.142058, 1.138581, 1.131171, 1.137090, 1.135874, 1.140580, 1.138066, 1.138240, 1.145058, 1.128826, 1.087457, 1.004262, 
        1.040773, 1.052561, 1.000048, 1.006319, 1.008273, 1.010241, 1.009674, 1.023909, 1.031341, 1.075399, 1.123558, 1.124310, 1.121368, 1.119877, 1.118510, 1.119339, 1.120187, 1.114476, 
        1.108833, 1.111823, 1.111339, 1.111884, 1.109921, 1.110546, 1.113972, 1.099747, 1.058709, 1.004262, 1.022560, 1.031240, 1.000048, 1.006319, 1.008273, 1.010241, 1.009674, 1.056049, 
        1.004187, 1.046779, 1.091419, 1.092190, 1.089649, 1.088015, 1.086245, 1.086323, 1.084537, 1.079597, 1.074338, 1.074555, 1.072413, 1.072353, 1.068120, 1.067509, 1.071156, 1.053541, 
        1.016809, 1.004262, 1.022560, 1.001873, 1.000048, 1.006319, 1.008273, 1.010241, 1.001310, 1.055563, 1.004187, 1.006212, 1.050846, 1.050423, 1.048931, 1.047128, 1.045364, 1.044571, 
        1.042123, 1.036002, 1.031195, 1.028148, 1.023533, 1.021181, 1.015756, 1.013636, 1.014241, 1.148380, 1.193790, 1.004262, 1.022560, 1.001873, 1.000048, 1.006319, 1.008273, 1.010241, 
        1.001310, 1.055563, 1.004187, 1.006212
    ])
# HF 1x1 scale factors will be a 5*12 array:
#  12 eta scale factors (30-41)
#  in 5 REAL ET bins (5, 20, 30, 50, Max)
#  So, index = etBin*12+ietaHF
caloStage2Params.layer1HFScaleETBins = cms.vint32([5, 20, 30, 50, 256])
# Old
# caloStage2Params.layer1HFScaleFactors = cms.vdouble([
#     1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 
#     1.767080, 1.767080, 1.755186, 1.769951, 1.763527, 1.791043, 1.898787, 1.982235, 2.071074, 2.193011, 2.356886, 2.403384, 
#     2.170477, 2.170477, 2.123540, 2.019866, 1.907698, 1.963179, 1.989122, 2.035251, 2.184642, 2.436399, 2.810884, 2.923750, 
#     1.943941, 1.943941, 1.899826, 1.813950, 1.714978, 1.736184, 1.785928, 1.834211, 1.944230, 2.153565, 2.720887, 2.749795, 
#     1.679984, 1.679984, 1.669753, 1.601871, 1.547276, 1.577805, 1.611497, 1.670184, 1.775022, 1.937061, 2.488311, 2.618629, 
#     ])
# Old multiplied by 0.7 (HF energies were formerly multiplied by 0.7; this has been removed)
caloStage2Params.layer1HFScaleFactors = cms.vdouble([
    1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
    1.236956, 1.236956, 1.228630, 1.238966, 1.234469, 1.253730, 1.329151, 1.387564, 1.449752, 1.535108, 1.649820, 1.682369,
    1.519334, 1.519334, 1.486478, 1.413906, 1.335389, 1.374225, 1.392385, 1.424676, 1.529249, 1.705479, 1.967619, 2.046625,
    1.360759, 1.360759, 1.329878, 1.269765, 1.200485, 1.215329, 1.250150, 1.283948, 1.360961, 1.507495, 1.904621, 1.924856,
    1.175989, 1.175989, 1.168827, 1.121310, 1.083093, 1.104463, 1.128048, 1.169129, 1.242515, 1.355943, 1.741818, 1.833040,
    ])
