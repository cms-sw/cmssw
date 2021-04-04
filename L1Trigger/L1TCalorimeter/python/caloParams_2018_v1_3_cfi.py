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
caloStage2Params.egEtaCut                   = cms.int32(28)
caloStage2Params.egSeedThreshold            = cms.double(2.)
caloStage2Params.egNeighbourThreshold       = cms.double(1.)
caloStage2Params.egHcalThreshold            = cms.double(0.)
caloStage2Params.egTrimmingLUTFile          = cms.FileInPath("L1Trigger/L1TCalorimeter/data/egTrimmingLUT_10_v16.01.19.txt")
caloStage2Params.egMaxHcalEt                = cms.double(0.)
caloStage2Params.egMaxPtHOverE          = cms.double(128.)
caloStage2Params.egMaxHOverELUTFile         = cms.FileInPath("L1Trigger/L1TCalorimeter/data/HoverEIdentification_0.995_v15.12.23.txt")
caloStage2Params.egHOverEcutBarrel          = cms.int32(3)
caloStage2Params.egHOverEcutEndcap          = cms.int32(4)
caloStage2Params.egBypassExtHOverE          = cms.uint32(0)
caloStage2Params.egCompressShapesLUTFile    = cms.FileInPath("L1Trigger/L1TCalorimeter/data/egCompressLUT_v4.txt")
caloStage2Params.egShapeIdType              = cms.string("compressed")
caloStage2Params.egShapeIdVersion           = cms.uint32(0)
caloStage2Params.egShapeIdLUTFile           = cms.FileInPath("L1Trigger/L1TCalorimeter/data/shapeIdentification_adapt0.99_compressedieta_compressedE_compressedshape_v15.12.08.txt")#Not used any more in the current emulator version, merged with calibration LUT

caloStage2Params.egPUSType                  = cms.string("None")
caloStage2Params.egIsolationType            = cms.string("compressed")
caloStage2Params.egIsoLUTFile               = cms.FileInPath("L1Trigger/L1TCalorimeter/data/EG_Iso_LUT_04_04_2017.2.txt")
caloStage2Params.egIsoLUTFile2               = cms.FileInPath("L1Trigger/L1TCalorimeter/data/EG_LoosestIso_2018.2.txt")
caloStage2Params.egIsoAreaNrTowersEta       = cms.uint32(2)
caloStage2Params.egIsoAreaNrTowersPhi       = cms.uint32(4)
caloStage2Params.egIsoVetoNrTowersPhi       = cms.uint32(2)
#caloStage2Params.egIsoPUEstTowerGranularity = cms.uint32(1)
#caloStage2Params.egIsoMaxEtaAbsForTowerSum  = cms.uint32(4)
#caloStage2Params.egIsoMaxEtaAbsForIsoSum    = cms.uint32(27)
caloStage2Params.egPUSParams                = cms.vdouble(1,4,32) #Isolation window in firmware goes up to abs(ieta)=32 for now
caloStage2Params.egCalibrationType          = cms.string("compressed")
caloStage2Params.egCalibrationVersion       = cms.uint32(0)
#caloStage2Params.egCalibrationLUTFile       = cms.FileInPath("L1Trigger/L1TCalorimeter/data/EG_Calibration_LUT_FW_v17.04.04_shapeIdentification_adapt0.99_compressedieta_compressedE_compressedshape_v15.12.08_correct.txt")
caloStage2Params.egCalibrationLUTFile       = cms.FileInPath("L1Trigger/L1TCalorimeter/data/corrections_Trimming10_compressedieta_compressedE_compressedshape_PANTELIS_v2_NEW_CALIBRATIONS_withShape_v17.04.04.txt")

# Tau
caloStage2Params.tauLsb                        = cms.double(0.5)
caloStage2Params.isoTauEtaMax                  = cms.int32(25)
caloStage2Params.tauSeedThreshold              = cms.double(0.)
caloStage2Params.tauNeighbourThreshold         = cms.double(0.)
caloStage2Params.tauIsoAreaNrTowersEta         = cms.uint32(2)
caloStage2Params.tauIsoAreaNrTowersPhi         = cms.uint32(4)
caloStage2Params.tauIsoVetoNrTowersPhi         = cms.uint32(2)
caloStage2Params.tauPUSType                    = cms.string("None")
caloStage2Params.tauIsoLUTFile                 = cms.FileInPath("L1Trigger/L1TCalorimeter/data/Tau_Iso_LUT_Option_31_extrap_2018_FW_v10.0.0.txt")
caloStage2Params.tauIsoLUTFile2                = cms.FileInPath("L1Trigger/L1TCalorimeter/data/Tau_Iso_LUT_Option_31_extrap_2018_FW_v10.0.0.txt")
#caloStage2Params.tauCalibrationLUTFile         = cms.FileInPath("L1Trigger/L1TCalorimeter/data/Tau_Calibration_LUT_2017_Layer1Calibration_FW_v12.0.0.txt")
caloStage2Params.tauCalibrationLUTFile         = cms.FileInPath("L1Trigger/L1TCalorimeter/data/Tau_Calibration_LUT_2018_Layer1CalibrationNewHCAL_FW_v13.0.0.txt")
caloStage2Params.tauCompressLUTFile            = cms.FileInPath("L1Trigger/L1TCalorimeter/data/tauCompressAllLUT_12bit_v3.txt")
caloStage2Params.tauPUSParams                  = cms.vdouble(1,4,32)

# jets
caloStage2Params.jetLsb                = cms.double(0.5)
caloStage2Params.jetSeedThreshold      = cms.double(4.0)
caloStage2Params.jetNeighbourThreshold = cms.double(0.)
caloStage2Params.jetPUSType            = cms.string("ChunkyDonut")
caloStage2Params.jetBypassPUS          = cms.uint32(0)

# Calibration options
caloStage2Params.jetCalibrationType = cms.string("LUT")

caloStage2Params.jetCompressPtLUTFile     = cms.FileInPath("L1Trigger/L1TCalorimeter/data/lut_pt_compress_2017v1.txt")
caloStage2Params.jetCompressEtaLUTFile    = cms.FileInPath("L1Trigger/L1TCalorimeter/data/lut_eta_compress_2017v1.txt")
caloStage2Params.jetCalibrationLUTFile    = cms.FileInPath("L1Trigger/L1TCalorimeter/data/lut_calib_2018v1_ECALZS_noHFJEC.txt")


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


# Layer 1 SF
caloStage2Params.layer1ECalScaleETBins = cms.vint32([3, 6, 9, 12, 15, 20, 25, 30, 35, 40, 45, 55, 70, 256])
caloStage2Params.layer1ECalScaleFactors = cms.vdouble([
        1.128436, 1.102229, 1.128385, 1.127897, 1.142444, 1.115476, 1.104283, 1.124583, 1.115929, 1.115196, 1.130342, 1.127173, 1.130640, 1.125474, 1.126652, 1.143535, 1.148905, 1.309035, 1.156021, 1.292685, 1.314302, 1.327634, 1.341229, 1.364885, 1.411117, 0.000000, 0.000000, 0.000000, 
        1.128436, 1.102229, 1.128385, 1.127897, 1.142444, 1.115476, 1.104283, 1.124583, 1.115929, 1.115196, 1.130342, 1.127173, 1.130640, 1.125474, 1.126652, 1.143535, 1.148905, 1.309035, 1.156021, 1.292685, 1.314302, 1.327634, 1.341229, 1.364885, 1.411117, 1.432419, 0.000000, 0.000000, 
        1.078545, 1.072734, 1.075464, 1.081920, 1.078434, 1.072281, 1.079780, 1.082043, 1.094741, 1.074544, 1.082784, 1.084089, 1.086375, 1.099718, 1.092858, 1.092855, 1.105166, 1.256155, 1.126301, 1.215671, 1.226302, 1.268900, 1.281721, 1.310629, 1.356976, 1.386428, 1.220159, 0.000000,  
        1.052366, 1.053986, 1.055250, 1.051033, 1.055017, 1.062249, 1.059624, 1.065355, 1.062623, 1.054089, 1.060477, 1.074504, 1.075570, 1.078549, 1.071588, 1.080279, 1.078463, 1.211087, 1.103915, 1.186517, 1.194161, 1.234868, 1.250080, 1.274639, 1.327394, 1.362218, 1.161404, 1.062366, 
        1.044640, 1.043507, 1.046185, 1.042067, 1.042425, 1.044121, 1.050677, 1.051604, 1.046070, 1.040140, 1.052732, 1.055652, 1.057201, 1.062982, 1.059512, 1.054542, 1.063873, 1.189094, 1.091948, 1.165298, 1.177338, 1.213632, 1.223587, 1.259376, 1.312025, 1.330172, 1.160220, 1.059058, 
        1.032947, 1.033877, 1.036016, 1.036056, 1.037819, 1.036489, 1.040341, 1.035373, 1.042736, 1.030510, 1.039291, 1.043943, 1.051946, 1.049653, 1.045154, 1.048874, 1.043392, 1.146608, 1.083743, 1.161479, 1.164940, 1.197187, 1.229915, 1.238886, 1.289410, 1.344620, 1.078591, 1.051894, 
        1.025813, 1.028301, 1.026054, 1.032050, 1.029899, 1.032383, 1.033763, 1.034211, 1.033892, 1.023902, 1.034960, 1.039866, 1.039984, 1.042478, 1.041047, 1.044143, 1.038748, 1.146814, 1.069148, 1.134356, 1.147952, 1.175102, 1.202532, 1.234549, 1.285897, 1.280056, 1.055845, 1.050155, 
        1.025370, 1.024465, 1.023378, 1.024989, 1.026322, 1.025140, 1.026122, 1.028451, 1.029161, 1.020083, 1.031555, 1.032971, 1.036222, 1.042410, 1.038053, 1.036796, 1.037195, 1.123576, 1.071556, 1.129229, 1.129561, 1.170449, 1.190240, 1.218357, 1.270482, 1.302586, 1.047321, 1.049100, 
        1.018591, 1.019825, 1.020823, 1.019265, 1.021761, 1.021521, 1.024053, 1.024121, 1.024979, 1.015315, 1.026035, 1.028734, 1.030409, 1.031414, 1.030694, 1.033450, 1.035642, 1.103688, 1.066969, 1.117955, 1.135950, 1.163170, 1.180714, 1.228736, 1.254963, 1.307361, 1.047123, 1.047264, 
        1.017483, 1.016714, 1.018925, 1.017087, 1.020438, 1.018852, 1.020796, 1.022534, 1.023495, 1.013378, 1.024097, 1.026067, 1.029037, 1.030731, 1.028759, 1.032480, 1.034680, 1.101491, 1.069770, 1.110644, 1.129222, 1.147881, 1.176695, 1.219110, 1.253033, 1.308691, 1.040706, 1.046607, 
        1.015432, 1.014445, 1.016057, 1.014908, 1.019115, 1.016567, 1.020411, 1.019852, 1.020255, 1.010779, 1.023433, 1.023674, 1.027479, 1.027385, 1.027332, 1.027537, 1.029061, 1.091079, 1.063278, 1.108876, 1.122727, 1.171282, 1.172058, 1.211259, 1.245839, 1.303968, 1.033863, 1.047743, 
        1.014370, 1.013304, 1.013397, 1.014261, 1.013673, 1.013183, 1.018534, 1.016581, 1.017015, 1.008220, 1.019515, 1.021560, 1.024502, 1.025611, 1.025905, 1.025863, 1.027252, 1.085230, 1.063040, 1.112256, 1.116617, 1.140393, 1.159214, 1.191434, 1.240601, 1.268525, 1.033247, 1.042853, 
        1.010174, 1.009843, 1.011520, 1.011041, 1.012957, 1.009075, 1.013178, 1.013301, 1.015033, 1.005133, 1.017533, 1.018564, 1.020319, 1.022634, 1.022429, 1.022338, 1.025613, 1.077639, 1.057895, 1.107098, 1.111157, 1.136106, 1.161737, 1.179259, 1.232736, 1.290141, 1.018941, 1.014733, 
        1.000302, 1.007651, 1.000751, 1.007791, 1.008949, 1.005394, 1.009599, 1.010180, 1.010865, 1.001827, 1.012447, 1.015231, 1.019545, 1.020611, 1.022404, 1.019032, 1.023113, 1.065127, 1.054688, 1.102754, 1.106151, 1.125574, 1.134480, 1.180965, 1.231939, 1.277289, 1.018941, 1.014733
    ])

caloStage2Params.layer1HCalScaleETBins = cms.vint32([6, 9, 12, 15, 20, 25, 30, 35, 40, 45, 55, 70, 256])
caloStage2Params.layer1HCalScaleFactors = cms.vdouble([
        1.691347, 1.704095, 1.729441, 1.735242, 1.726367, 1.780424, 1.794996, 1.815904, 1.817388, 1.894632, 1.932656, 1.957527, 1.970890, 2.005818, 2.041546, 2.042775, 1.989288, 1.594904, 
        1.659821, 1.676038, 1.495936, 1.505035, 1.512590, 1.511470, 1.494893, 1.378435, 1.430994, 1.500227, 1.531796, 1.547539, 1.559295, 1.561478, 1.568922, 1.601485, 1.616591, 1.620739, 
        1.642884, 1.678420, 1.692987, 1.728681, 1.728957, 1.766650, 1.782739, 1.782875, 1.751371, 1.431918, 1.487225, 1.483881, 1.336485, 1.349895, 1.363924, 1.375728, 1.377818, 1.310078, 
        1.334588, 1.399686, 1.465418, 1.462800, 1.475840, 1.474735, 1.474407, 1.506928, 1.526279, 1.524000, 1.532718, 1.583398, 1.608380, 1.623528, 1.619634, 1.646501, 1.667856, 1.674628, 
        1.635381, 1.350235, 1.394938, 1.383940, 1.244552, 1.256971, 1.261180, 1.282746, 1.279512, 1.221092, 1.241831, 1.351526, 1.390201, 1.404198, 1.416259, 1.404045, 1.418265, 1.437914, 
        1.450857, 1.463511, 1.462653, 1.501891, 1.518896, 1.548252, 1.545831, 1.565901, 1.574314, 1.575115, 1.557629, 1.301893, 1.326949, 1.312526, 1.197573, 1.210304, 1.222283, 1.239081, 
        1.240673, 1.185591, 1.207651, 1.275166, 1.314260, 1.335228, 1.340603, 1.323027, 1.324793, 1.347954, 1.349916, 1.363145, 1.359628, 1.402624, 1.416518, 1.457202, 1.461053, 1.484090, 
        1.500787, 1.498450, 1.471731, 1.215732, 1.253565, 1.243598, 1.157168, 1.164428, 1.175435, 1.189310, 1.192682, 1.142038, 1.162810, 1.230426, 1.262901, 1.265380, 1.274364, 1.276111, 
        1.282349, 1.291748, 1.305521, 1.301818, 1.305124, 1.336506, 1.345742, 1.357458, 1.370139, 1.381995, 1.394554, 1.388952, 1.363805, 1.166810, 1.204780, 1.193913, 1.118331, 1.124657, 
        1.136138, 1.148564, 1.147392, 1.085564, 1.109949, 1.184837, 1.221607, 1.219692, 1.235950, 1.230444, 1.234908, 1.245100, 1.256813, 1.252608, 1.263569, 1.284188, 1.300083, 1.309901, 
        1.312849, 1.335500, 1.339967, 1.328269, 1.309282, 1.128239, 1.173002, 1.163030, 1.077388, 1.087037, 1.085620, 1.099773, 1.097418, 1.047416, 1.080447, 1.135984, 1.186335, 1.189457, 
        1.186903, 1.191054, 1.192951, 1.218812, 1.222226, 1.220196, 1.221331, 1.264243, 1.284869, 1.277098, 1.263366, 1.276293, 1.291829, 1.275918, 1.248086, 1.095700, 1.143874, 1.132783, 
        1.054939, 1.055922, 1.055405, 1.058330, 1.062463, 1.012972, 1.028538, 1.089975, 1.155949, 1.153120, 1.157186, 1.163320, 1.157607, 1.174722, 1.181157, 1.179473, 1.186948, 1.192614, 
        1.207973, 1.215075, 1.252322, 1.231549, 1.241483, 1.224214, 1.207592, 1.069829, 1.112551, 1.107158, 1.025349, 1.026181, 1.028466, 1.035129, 1.030918, 0.977843, 1.004295, 1.075236, 
        1.122942, 1.124839, 1.130900, 1.139241, 1.134602, 1.141732, 1.154381, 1.154366, 1.162207, 1.167863, 1.182334, 1.189497, 1.179567, 1.185553, 1.205978, 1.188532, 1.154839, 1.058371, 
        1.096597, 1.086545, 0.997724, 1.000690, 1.005683, 1.009107, 1.006028, 0.962736, 0.974019, 1.035748, 1.094997, 1.098600, 1.101567, 1.102895, 1.106445, 1.113255, 1.114956, 1.118930, 
        1.128154, 1.135288, 1.145308, 1.151612, 1.142554, 1.153640, 1.154025, 1.138100, 1.127446, 1.034945, 1.069153, 1.062188, 0.977909, 0.972598, 0.972539, 0.978454, 0.975065, 0.941113, 
        0.948722, 1.004971, 1.055020, 1.054883, 1.059317, 1.061911, 1.062005, 1.066707, 1.074156, 1.064278, 1.072810, 1.076579, 1.084072, 1.091055, 1.090640, 1.086634, 1.095179, 1.075771, 
        1.051884, 1.005930, 1.033331, 1.024734, 0.943637, 0.941986, 0.937779, 0.943865, 0.928477, 0.902234, 0.908232, 0.960607, 1.005841, 1.011405, 1.012527, 1.015557, 1.014508, 1.020877, 
        1.019076, 1.015173, 1.015651, 1.019594, 1.026845, 1.024959, 1.025915, 1.029455, 1.017985, 1.016933, 0.989723, 0.977768, 0.993744, 0.985200, 0.907247, 0.903328, 0.912164, 0.898908, 
        0.886431, 0.851162, 0.863541, 0.890523
    ])

caloStage2Params.layer1HFScaleETBins = cms.vint32([6, 9, 12, 15, 20, 25, 30, 35, 40, 45, 55, 70, 256])

caloStage2Params.layer1HFScaleFactors = cms.vdouble([
    1.401648, 1.138462, 1.188641, 1.173580, 1.218745, 1.238716, 1.279351, 1.306353, 1.352201, 1.425513, 1.766435, 1.913788, 
    1.284459, 1.081785, 1.142576, 1.136715, 1.118770, 1.156336, 1.173606, 1.201886, 1.276473, 1.291960, 1.663819, 1.755983, 
    1.204911, 1.055869, 1.092000, 1.100584, 1.049183, 1.082497, 1.082629, 1.113835, 1.167107, 1.195128, 1.573579, 1.696511, 
    1.148313, 1.036986, 1.094283, 1.054889, 1.001794, 1.023684, 1.032747, 1.048416, 1.096110, 1.146655, 1.541076, 1.641335, 
    1.100579, 1.014459, 1.045712, 1.024446, 0.964075, 0.978100, 0.990507, 1.007612, 1.048333, 1.095952, 1.501771, 1.598874, 
    1.062801, 0.988340, 1.024694, 0.983528, 0.925067, 0.939920, 0.957879, 0.972375, 1.007824, 1.060773, 1.440472, 1.545380, 
    1.022108, 0.970880, 1.008919, 0.956236, 0.907449, 0.922011, 0.938728, 0.950612, 0.987919, 1.036760, 1.395258, 1.500256, 
    0.997997, 0.952793, 0.987330, 0.928038, 0.891417, 0.906300, 0.921343, 0.936786, 0.970373, 1.014640, 1.342585, 1.464742, 
    0.980560, 0.939970, 0.976677, 0.911275, 0.883970, 0.896411, 0.909605, 0.928250, 0.960889, 1.000147, 1.299124, 1.444636, 
    0.960950, 0.924990, 0.949128, 0.902849, 0.880957, 0.892083, 0.904219, 0.920509, 0.950512, 0.989206, 1.253466, 1.428955, 
    0.947099, 0.903350, 0.941550, 0.892063, 0.873377, 0.886352, 0.900072, 0.915840, 0.944985, 0.981827, 1.199814, 1.410339, 
    0.915085, 0.901126, 0.930501, 0.883655, 0.869623, 0.883995, 0.896408, 0.913637, 0.946368, 0.979263, 1.145768, 1.360238, 
    0.886918, 0.895145, 0.914478, 0.882066, 0.871161, 0.886831, 0.900315, 0.917568, 0.952193, 0.984897, 1.097738, 1.285041
    ])
