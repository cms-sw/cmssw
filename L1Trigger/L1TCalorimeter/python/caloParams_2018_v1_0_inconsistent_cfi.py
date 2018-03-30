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

caloStage2Params.etSumMetPUSLUTFile               = cms.FileInPath("L1Trigger/L1TCalorimeter/data/lut_towEtThresh_2017v7.txt")
caloStage2Params.etSumEttPUSLUTFile               = cms.FileInPath("L1Trigger/L1TCalorimeter/data/lut_towEtThresh_dummy.txt")
caloStage2Params.etSumEcalSumPUSLUTFile           = cms.FileInPath("L1Trigger/L1TCalorimeter/data/lut_towEtThresh_dummy.txt")
caloStage2Params.etSumXCalibrationLUTFile         = cms.FileInPath("L1Trigger/L1TCalorimeter/data/lut_etSumPUS_dummy.txt")
caloStage2Params.etSumYCalibrationLUTFile         = cms.FileInPath("L1Trigger/L1TCalorimeter/data/lut_etSumPUS_dummy.txt")
caloStage2Params.etSumEttCalibrationLUTFile       = cms.FileInPath("L1Trigger/L1TCalorimeter/data/lut_etSumPUS_dummy.txt")
caloStage2Params.etSumEcalSumCalibrationLUTFile   = cms.FileInPath("L1Trigger/L1TCalorimeter/data/lut_etSumPUS_dummy.txt")


caloStage2Params.layer1ECalScaleETBins = cms.vint32([6, 9, 12, 15, 20, 25, 30, 35, 40, 45, 55, 70, 256])
caloStage2Params.layer1ECalScaleFactors = cms.vdouble([
        1.128436, 1.102229, 1.128385, 1.127897, 1.142444, 1.115476, 1.104283, 1.124583, 1.115929, 1.115196, 1.130342, 1.127173, 1.130640, 1.125474, 1.126652, 1.143535, 
        1.148905, 1.309035, 1.156021, 1.292685, 1.314302, 1.327634, 1.341229, 1.364885, 1.411117, 1.432419, 1.288526, 1.082139, 1.078545, 1.072734, 1.075464, 1.081920, 1.078434, 1.072281, 
        1.079780, 1.082043, 1.094741, 1.074544, 1.082784, 1.084089, 1.086375, 1.099718, 1.092858, 1.092855, 1.105166, 1.256155, 1.126301, 1.215671, 1.226302, 1.268900, 1.281721, 1.310629, 
        1.356976, 1.386428, 1.220159, 1.066925, 1.052366, 1.053986, 1.055250, 1.051033, 1.055017, 1.062249, 1.059624, 1.065355, 1.062623, 1.054089, 1.060477, 1.074504, 1.075570, 1.078549, 
        1.071588, 1.080279, 1.078463, 1.211087, 1.103915, 1.186517, 1.194161, 1.234868, 1.250080, 1.274639, 1.327394, 1.362218, 1.161404, 1.062366, 1.044640, 1.043507, 1.046185, 1.042067, 
        1.042425, 1.044121, 1.050677, 1.051604, 1.046070, 1.040140, 1.052732, 1.055652, 1.057201, 1.062982, 1.059512, 1.054542, 1.063873, 1.189094, 1.091948, 1.165298, 1.177338, 1.213632, 
        1.223587, 1.259376, 1.312025, 1.330172, 1.160220, 1.059058, 1.032947, 1.033877, 1.036016, 1.036056, 1.037819, 1.036489, 1.040341, 1.035373, 1.042736, 1.030510, 1.039291, 1.043943, 
        1.051946, 1.049653, 1.045154, 1.048874, 1.043392, 1.146608, 1.083743, 1.161479, 1.164940, 1.197187, 1.229915, 1.238886, 1.289410, 1.344620, 1.078591, 1.051894, 1.025813, 1.028301, 
        1.026054, 1.032050, 1.029899, 1.032383, 1.033763, 1.034211, 1.033892, 1.023902, 1.034960, 1.039866, 1.039984, 1.042478, 1.041047, 1.044143, 1.038748, 1.146814, 1.069148, 1.134356, 
        1.147952, 1.175102, 1.202532, 1.234549, 1.285897, 1.280056, 1.055845, 1.050155, 1.025370, 1.024465, 1.023378, 1.024989, 1.026322, 1.025140, 1.026122, 1.028451, 1.029161, 1.020083, 
        1.031555, 1.032971, 1.036222, 1.042410, 1.038053, 1.036796, 1.037195, 1.123576, 1.071556, 1.129229, 1.129561, 1.170449, 1.190240, 1.218357, 1.270482, 1.302586, 1.047321, 1.049100, 
        1.018591, 1.019825, 1.020823, 1.019265, 1.021761, 1.021521, 1.024053, 1.024121, 1.024979, 1.015315, 1.026035, 1.028734, 1.030409, 1.031414, 1.030694, 1.033450, 1.035642, 1.103688, 
        1.066969, 1.117955, 1.135950, 1.163170, 1.180714, 1.228736, 1.254963, 1.307361, 1.047123, 1.047264, 1.017483, 1.016714, 1.018925, 1.017087, 1.020438, 1.018852, 1.020796, 1.022534, 
        1.023495, 1.013378, 1.024097, 1.026067, 1.029037, 1.030731, 1.028759, 1.032480, 1.034680, 1.101491, 1.069770, 1.110644, 1.129222, 1.147881, 1.176695, 1.219110, 1.253033, 1.308691, 
        1.040706, 1.046607, 1.015432, 1.014445, 1.016057, 1.014908, 1.019115, 1.016567, 1.020411, 1.019852, 1.020255, 1.010779, 1.023433, 1.023674, 1.027479, 1.027385, 1.027332, 1.027537, 
        1.029061, 1.091079, 1.063278, 1.108876, 1.122727, 1.171282, 1.172058, 1.211259, 1.245839, 1.303968, 1.033863, 1.047743, 1.014370, 1.013304, 1.013397, 1.014261, 1.013673, 1.013183, 
        1.018534, 1.016581, 1.017015, 1.008220, 1.019515, 1.021560, 1.024502, 1.025611, 1.025905, 1.025863, 1.027252, 1.085230, 1.063040, 1.112256, 1.116617, 1.140393, 1.159214, 1.191434, 
        1.240601, 1.268525, 1.033247, 1.042853, 1.010174, 1.009843, 1.011520, 1.011041, 1.012957, 1.009075, 1.013178, 1.013301, 1.015033, 1.005133, 1.017533, 1.018564, 1.020319, 1.022634, 
        1.022429, 1.022338, 1.025613, 1.077639, 1.057895, 1.107098, 1.111157, 1.136106, 1.161737, 1.179259, 1.232736, 1.290141, 1.018941, 1.014733, 1.000302, 1.007651, 1.000751, 1.007791, 
        1.008949, 1.005394, 1.009599, 1.010180, 1.010865, 1.001827, 1.012447, 1.015231, 1.019545, 1.020611, 1.022404, 1.019032, 1.023113, 1.065127, 1.054688, 1.102754, 1.106151, 1.125574, 
        1.134480, 1.180965, 1.231939, 1.277289, 1.018941, 1.014733
    ])
        
caloStage2Params.layer1HCalScaleETBins = cms.vint32([6, 9, 12, 15, 20, 25, 30, 35, 40, 45, 55, 70, 256])
caloStage2Params.layer1HCalScaleFactors = cms.vdouble([
        1.262598, 1.263451, 1.264363, 1.268886, 1.272013, 1.286281, 1.287660, 1.292219, 1.298271, 1.320577, 1.315319, 1.317451, 1.328640, 1.338891, 1.329242, 1.347992, 1.355456, 1.243778, 
        1.235876, 1.238732, 1.224452, 1.226103, 1.228549, 1.225896, 1.216829, 1.164620, 1.189367, 1.173541, 1.246936, 1.243398, 1.244548, 1.247337, 1.250756, 1.267811, 1.265451, 1.269533, 
        1.278960, 1.302687, 1.292846, 1.297518, 1.307792, 1.316213, 1.304653, 1.323715, 1.322030, 1.215977, 1.212917, 1.216943, 1.190625, 1.194127, 1.198234, 1.198832, 1.193084, 1.157383, 
        1.180753, 1.164592, 1.226835, 1.224542, 1.227072, 1.226591, 1.231005, 1.246072, 1.248907, 1.250343, 1.255004, 1.276186, 1.276042, 1.275162, 1.287275, 1.296967, 1.290800, 1.303941, 
        1.302449, 1.194237, 1.194299, 1.196477, 1.162766, 1.169363, 1.169994, 1.174923, 1.171337, 1.121800, 1.161657, 1.157112, 1.204370, 1.209401, 1.212756, 1.212091, 1.215896, 1.226607, 
        1.233116, 1.230661, 1.242397, 1.255865, 1.256891, 1.262414, 1.270640, 1.279181, 1.274226, 1.280737, 1.276476, 1.175549, 1.178926, 1.175063, 1.147737, 1.148297, 1.142264, 1.158086, 
        1.151300, 1.123937, 1.136349, 1.148170, 1.189364, 1.188098, 1.190765, 1.191948, 1.193431, 1.207394, 1.206893, 1.207988, 1.215455, 1.235275, 1.233729, 1.237088, 1.244701, 1.253612, 
        1.250603, 1.258984, 1.251955, 1.153844, 1.198514, 1.156658, 1.111885, 1.116923, 1.120125, 1.121531, 1.109889, 1.074981, 1.109417, 1.133790, 1.166172, 1.165788, 1.168602, 1.170184, 
        1.170998, 1.182646, 1.185352, 1.185812, 1.185921, 1.209223, 1.204945, 1.210594, 1.215852, 1.230937, 1.225605, 1.228218, 1.217748, 1.130660, 1.133770, 1.131676, 1.085194, 1.090145, 
        1.089829, 1.089889, 1.095883, 1.048375, 1.067871, 1.115329, 1.143850, 1.144430, 1.149566, 1.149868, 1.151144, 1.162380, 1.164385, 1.164053, 1.163972, 1.181097, 1.187641, 1.187835, 
        1.202350, 1.208069, 1.197829, 1.200694, 1.194265, 1.109278, 1.114857, 1.120655, 1.047812, 1.065302, 1.059292, 1.064588, 1.064599, 1.028215, 1.050673, 1.097516, 1.125161, 1.127613, 
        1.127460, 1.129376, 1.129858, 1.144006, 1.144816, 1.144215, 1.155880, 1.166303, 1.162317, 1.168225, 1.183289, 1.175743, 1.175979, 1.177247, 1.158111, 1.087837, 1.097769, 1.094044, 
        1.035324, 1.036461, 1.039377, 1.036659, 1.044379, 1.043209, 1.025333, 1.075897, 1.108214, 1.108518, 1.107588, 1.113938, 1.109980, 1.122504, 1.126634, 1.119655, 1.128499, 1.132332, 
        1.133671, 1.146619, 1.146455, 1.153922, 1.162871, 1.146704, 1.134645, 1.071514, 1.081605, 1.073939, 1.011888, 1.019214, 1.012181, 1.023941, 1.002201, 1.000000, 1.000000, 1.053242, 
        1.086406, 1.092709, 1.094821, 1.099021, 1.096816, 1.102261, 1.108524, 1.107258, 1.115337, 1.113792, 1.125827, 1.128253, 1.126608, 1.134247, 1.133597, 1.126248, 1.112667, 1.059377, 
        1.069952, 1.060658, 1.000000, 1.001808, 1.000744, 1.001474, 1.000815, 1.000000, 1.000000, 1.036238, 1.070718, 1.072674, 1.075761, 1.077347, 1.077116, 1.083908, 1.085992, 1.086417, 
        1.095160, 1.094859, 1.098117, 1.098375, 1.095923, 1.110562, 1.117002, 1.099837, 1.079888, 1.040424, 1.042538, 1.038363, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 
        1.000000, 1.002540, 1.042308, 1.043274, 1.046904, 1.048218, 1.048620, 1.053754, 1.057211, 1.052521, 1.058235, 1.059201, 1.035636, 1.066950, 1.063419, 1.076206, 1.067340, 1.062315, 
        1.050958, 1.014276, 1.025376, 1.015995, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.003310, 1.008922, 1.009786, 1.011583, 1.010195, 1.016797, 
        1.015394, 1.013312, 1.020909, 1.015590, 1.015764, 1.016447, 1.013844, 1.015128, 1.030743, 1.005025, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 
        1.000000, 1.000000, 1.000000, 1.000000
    ])
        

caloStage2Params.layer1HFScaleETBins = cms.vint32([6, 9, 12, 15, 20, 25, 30, 35, 40, 45, 55, 70, 256])

caloStage2Params.layer1HFScaleFactors = cms.vdouble([
    1.401648, 1.138462, 1.188641, 1.173580, 1.218745, 1.238716, 1.279351, 1.306353, 1.352201, 1.425513, 1.766435, 1.913788, 
    1.284459, 1.081785, 1.142576, 1.136715, 1.118770, 1.156336, 1.173606, 1.201886, 1.276473, 1.291960, 1.663819, 1.755983, 
    1.204911, 1.055869, 1.092000, 1.100584, 1.049183, 1.082497, 1.082629, 1.113835, 1.167107, 1.195128, 1.573579, 1.696511, 
    1.148313, 1.036986, 1.094283, 1.054889, 1.001794, 1.023684, 1.032747, 1.048416, 1.096110, 1.146655, 1.541076, 1.641335, 
    1.100579, 1.014459, 1.045712, 1.024446, 1.000000, 1.000000, 1.000000, 1.007612, 1.048333, 1.095952, 1.501771, 1.598874, 
    1.062801, 1.000000, 1.024694, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.007824, 1.060773, 1.440472, 1.545380, 
    1.022108, 1.000000, 1.008919, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.036760, 1.395258, 1.500256, 
    1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.014640, 1.342585, 1.464742, 
    1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000147, 1.299124, 1.444636, 
    1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.253466, 1.428955, 
    1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.199814, 1.410339, 
    1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.145768, 1.360238, 
    1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.097738, 1.285041
    ])

    
