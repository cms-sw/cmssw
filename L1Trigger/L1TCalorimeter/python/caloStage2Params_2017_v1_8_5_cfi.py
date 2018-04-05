#
# caloStage2Params_2017_v1_8_5_cfi
# change w.r.t. v1_8_4: tightened H/E cut in the endcaps for L1EG, barrel unchanged wrt default
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
caloStage2Params.egEtaCut                   = cms.int32(28)
caloStage2Params.egSeedThreshold            = cms.double(2.)
caloStage2Params.egNeighbourThreshold       = cms.double(1.)
caloStage2Params.egHcalThreshold            = cms.double(0.)
caloStage2Params.egTrimmingLUTFile          = cms.FileInPath("L1Trigger/L1TCalorimeter/data/egTrimmingLUT_10_v16.01.19.txt")
caloStage2Params.egMaxHcalEt                = cms.double(0.)
caloStage2Params.egMaxPtHOverE          = cms.double(128.)
caloStage2Params.egMaxHOverELUTFile         = cms.FileInPath("L1Trigger/L1TCalorimeter/data/HoverEIdentification_0.995_v15.12.23.txt")
caloStage2Params.egHOverEcutBarrel          = cms.int32(4)
caloStage2Params.egHOverEcutEndcap          = cms.int32(4)
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
caloStage2Params.egPUSParams                = cms.vdouble(1,4,32) #Isolation window in firmware goes up to abs(ieta)=32 for now
caloStage2Params.egCalibrationType          = cms.string("compressed")
caloStage2Params.egCalibrationVersion       = cms.uint32(0)
caloStage2Params.egCalibrationLUTFile       = cms.FileInPath("L1Trigger/L1TCalorimeter/data/EG_Calibration_LUT_FW_v17.04.04_shapeIdentification_adapt0.99_compressedieta_compressedE_compressedshape_v15.12.08_correct.txt")


# Tau
caloStage2Params.tauLsb                        = cms.double(0.5)
caloStage2Params.isoTauEtaMax                  = cms.int32(25)
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


# Layer 1 LUT specification
#
# Et-dependent scale factors
# ECal/HCal scale factors will be a 9*28 array:
#   28 eta scale factors (1-28)
#   in 9 ET bins (10, 15, 20, 25, 30, 35, 40, 45, Max)
#  So, index = etBin*28+ieta
#FInal ecal and HCAl calibrations using mean.. 
#caloStage2Params.layer1ECalScaleETBins = cms.vint32([10, 15, 20, 25, 30, 35, 40, 45, 256])
caloStage2Params.layer1ECalScaleETBins = cms.vint32([6, 9, 12, 15, 20, 25, 30, 35, 40, 45, 55, 70, 256])
caloStage2Params.layer1ECalScaleFactors = cms.vdouble([
        1.230770, 1.208991, 1.215351, 1.213350, 1.208275, 1.219069, 1.222985, 1.217729, 1.221262, 1.238302, 1.251040, 1.263300, 1.323678, 1.330580, 1.326592, 1.348387, 1.444263, 
        2.278846, 1.519648, 1.646930, 1.655937, 1.632773, 1.599511, 1.646612, 1.654494, 1.617013, 1.486696, 1.509178, 1.200818, 1.188237, 1.183589, 1.208003, 1.181402, 1.225384, 
        1.225197, 1.238568, 1.229019, 1.226685, 1.247974, 1.243772, 1.247896, 1.307835, 1.292458, 1.292431, 1.408326, 2.086560, 1.375806, 1.479160, 1.471558, 1.472672, 1.470455, 1.504425, 
        1.522293, 1.505404, 1.396942, 1.439355, 1.157450, 1.163603, 1.172625, 1.215935, 1.164385, 1.192617, 1.175256, 1.189703, 1.194979, 1.211745, 1.201289, 1.213436, 1.233816, 1.263384, 
        1.261724, 1.270097, 1.355681, 1.948426, 1.289409, 1.380269, 1.429089, 1.409515, 1.402992, 1.432428, 1.454425, 1.448494, 1.356177, 1.415035, 1.109329, 1.130229, 1.136506, 1.156603, 
        1.148180, 1.136802, 1.157053, 1.146116, 1.147593, 1.160180, 1.157040, 1.175752, 1.176997, 1.192450, 1.192074, 1.219774, 1.279824, 1.946019, 1.262478, 1.341427, 1.385701, 1.349399, 
        1.344777, 1.377678, 1.428243, 1.404236, 1.338994, 1.396023, 1.089770, 1.082103, 1.090228, 1.089163, 1.090443, 1.103950, 1.093130, 1.097591, 1.112827, 1.115258, 1.115559, 1.119360, 
        1.139471, 1.152378, 1.154278, 1.156148, 1.206562, 1.852929, 1.225639, 1.299721, 1.351968, 1.317715, 1.315265, 1.349087, 1.395426, 1.382078, 1.313055, 1.375904, 1.071911, 1.067929, 
        1.066290, 1.064308, 1.070857, 1.077186, 1.072581, 1.073245, 1.086873, 1.104813, 1.080402, 1.086703, 1.096386, 1.114866, 1.122498, 1.129279, 1.167211, 1.815989, 1.202916, 1.279602, 
        1.328353, 1.295723, 1.289288, 1.324630, 1.370921, 1.360803, 1.295568, 1.360794, 1.059307, 1.055444, 1.055547, 1.054844, 1.055633, 1.070619, 1.067176, 1.060437, 1.064041, 1.082473, 
        1.077814, 1.076358, 1.083088, 1.096413, 1.110072, 1.117249, 1.141054, 1.656952, 1.183193, 1.250519, 1.271871, 1.264930, 1.265938, 1.305469, 1.346518, 1.333812, 1.280760, 1.349732, 
        1.050145, 1.045243, 1.048544, 1.045844, 1.048177, 1.065325, 1.054551, 1.058521, 1.054766, 1.072195, 1.064060, 1.066467, 1.077743, 1.085108, 1.080674, 1.098568, 1.129347, 1.505618, 
        1.177574, 1.230855, 1.230263, 1.254251, 1.260679, 1.296709, 1.339361, 1.331497, 1.276333, 1.346620, 1.044434, 1.046443, 1.044523, 1.045117, 1.042187, 1.054454, 1.047753, 1.049621, 
        1.054001, 1.067735, 1.058934, 1.056480, 1.067558, 1.092601, 1.075606, 1.082969, 1.119556, 1.487269, 1.163707, 1.220454, 1.211224, 1.245848, 1.245011, 1.276404, 1.320622, 1.317634, 
        1.259293, 1.331703, 1.036507, 1.031687, 1.034341, 1.038718, 1.039960, 1.056759, 1.042963, 1.047073, 1.046287, 1.061804, 1.047322, 1.051865, 1.061811, 1.077710, 1.065041, 1.074170, 
        1.110572, 1.409428, 1.161587, 1.209903, 1.200401, 1.237205, 1.239067, 1.274367, 1.311375, 1.309819, 1.206232, 1.281656, 1.034128, 1.028977, 1.031212, 1.032869, 1.032576, 1.048159, 
        1.035764, 1.037963, 1.038160, 1.051300, 1.042707, 1.043292, 1.051796, 1.064664, 1.058869, 1.066773, 1.098222, 1.340830, 1.150792, 1.192509, 1.188802, 1.222087, 1.222194, 1.261393, 
        1.296707, 1.293975, 1.098204, 1.090501, 1.024842, 1.022760, 1.023820, 1.023853, 1.025343, 1.037737, 1.029393, 1.028506, 1.030900, 1.041196, 1.031898, 1.034541, 1.040246, 1.054993, 
        1.048250, 1.054786, 1.080644, 1.270714, 1.140567, 1.177924, 1.174622, 1.207597, 1.215113, 1.250065, 1.286407, 1.284189, 1.284189, 1.284189, 1.018863, 1.016991, 1.017559, 1.017933, 
        1.019889, 1.027862, 1.020909, 1.022602, 1.024068, 1.031407, 1.025612, 1.027150, 1.033400, 1.043217, 1.039784, 1.044029, 1.063090, 1.203727, 1.127824, 1.157782, 1.153493, 1.179254, 
        1.181702, 1.203665, 1.236615, 1.223496, 1.223496, 1.223496,
    ])
caloStage2Params.layer1HCalScaleETBins = cms.vint32([6, 9, 12, 15, 20, 25, 30, 35, 40, 45, 55, 70, 256])
caloStage2Params.layer1HCalScaleFactors = cms.vdouble([
        1.461291, 1.454625, 1.455598, 1.449939, 1.460976, 1.473480, 1.466595, 1.468804, 1.481637, 1.497836, 1.504469, 1.517996, 1.543596, 1.558518, 1.575825, 1.586335, 1.544925, 
        1.389349, 1.396040, 1.439915, 1.398678, 1.441202, 1.451638, 1.471108, 1.491032, 1.530464, 1.654074, 1.650121, 1.375991, 1.376866, 1.374574, 1.372501, 1.375971, 1.388565, 
        1.389183, 1.393546, 1.396302, 1.418089, 1.422244, 1.443768, 1.457298, 1.469873, 1.485083, 1.488035, 1.443898, 1.288701, 1.318062, 1.354571, 1.296165, 1.316493, 1.324791, 
        1.329227, 1.329808, 1.347197, 1.385654, 1.367404, 1.330324, 1.330662, 1.332125, 1.334518, 1.330010, 1.343405, 1.346451, 1.346908, 1.349793, 1.374879, 1.374081, 1.391796, 
        1.406290, 1.417394, 1.433500, 1.422528, 1.373563, 1.235496, 1.274484, 1.300781, 1.222220, 1.234458, 1.242407, 1.238218, 1.231158, 1.229238, 1.248514, 1.261873, 1.301111, 
        1.303505, 1.304342, 1.300883, 1.301955, 1.312233, 1.316385, 1.317202, 1.315308, 1.335695, 1.342003, 1.353083, 1.362459, 1.370255, 1.379031, 1.369051, 1.319807, 1.199383, 
        1.235890, 1.263149, 1.172396, 1.189026, 1.182243, 1.174765, 1.160341, 1.159067, 1.211505, 1.281201, 1.275768, 1.275028, 1.274251, 1.272298, 1.272090, 1.283374, 1.284263, 
        1.283754, 1.283420, 1.301406, 1.301963, 1.309314, 1.315140, 1.319884, 1.327534, 1.312294, 1.264995, 1.168509, 1.199964, 1.223081, 1.121649, 1.128651, 1.126169, 1.112205, 
        1.112272, 1.146819, 1.194809, 1.195220, 1.247213, 1.246675, 1.246937, 1.246958, 1.242936, 1.249726, 1.251084, 1.250033, 1.246985, 1.258533, 1.259635, 1.264881, 1.266468, 
        1.266983, 1.271018, 1.255678, 1.213176, 1.138423, 1.164712, 1.181474, 1.075510, 1.076280, 1.074546, 1.080937, 1.122946, 1.131501, 1.111036, 1.110220, 1.225062, 1.224108, 
        1.223705, 1.222218, 1.218175, 1.223123, 1.224592, 1.222768, 1.215725, 1.224004, 1.223652, 1.225737, 1.227469, 1.227681, 1.230391, 1.213227, 1.171645, 1.115248, 1.135580, 
        1.145603, 1.037666, 1.051260, 1.080362, 1.099305, 1.088466, 1.075173, 1.056630, 1.032875, 1.202830, 1.204155, 1.203486, 1.201967, 1.194002, 1.197947, 1.198415, 1.196705, 
        1.188166, 1.196680, 1.192931, 1.197918, 1.197352, 1.192337, 1.193319, 1.177775, 1.139192, 1.096009, 1.110544, 1.116609, 1.042172, 1.074235, 1.080403, 1.064799, 1.053359, 
        1.037543, 1.006759, 1.006759, 1.188028, 1.183705, 1.183068, 1.180943, 1.176596, 1.180435, 1.176822, 1.172566, 1.167376, 1.172652, 1.171996, 1.172278, 1.170926, 1.164278, 
        1.161894, 1.142646, 1.108552, 1.076311, 1.086825, 1.092867, 1.057110, 1.060453, 1.051915, 1.037153, 1.022450, 1.001390, 1.001390, 1.001390, 1.169376, 1.169865, 1.167855, 
        1.166754, 1.159677, 1.161727, 1.161602, 1.159157, 1.149591, 1.155508, 1.150864, 1.150443, 1.147936, 1.137900, 1.131026, 1.112978, 1.080918, 1.061229, 1.066055, 1.073614, 
        1.042392, 1.038032, 1.029904, 1.012558, 1.012558, 1.012558, 1.012558, 1.012558, 1.150175, 1.150513, 1.150216, 1.147693, 1.140891, 1.143654, 1.140204, 1.134718, 1.128271, 
        1.127001, 1.122190, 1.119082, 1.114472, 1.102259, 1.089926, 1.075835, 1.045855, 1.041539, 1.044984, 1.046205, 1.015383, 1.011540, 1.001439, 1.001439, 1.001439, 1.001439, 
        1.001439, 1.001439, 1.125661, 1.124191, 1.123250, 1.118812, 1.112736, 1.111236, 1.106057, 1.098389, 1.089647, 1.086947, 1.079851, 1.073968, 1.068117, 1.051795, 1.035960, 
        1.021170, 1.001873, 1.014026, 1.014703, 1.023760, 1.023760, 1.023760, 1.023760, 1.023760, 1.023760, 1.023760, 1.023760, 1.023760, 1.090500, 1.089670, 1.086302, 1.082156, 
        1.075398, 1.072158, 1.065357, 1.056707, 1.048909, 1.041525, 1.033163, 1.023515, 1.015901, 1.015901, 1.015901, 1.137542, 1.188167, 1.019230, 1.019230, 1.036922, 1.036922, 
        1.036922, 1.036922, 1.036922, 1.036922, 1.036922, 1.036922, 1.036922,
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
    0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7,
    1.236956, 1.236956, 1.228630, 1.238966, 1.234469, 1.253730, 1.329151, 1.387564, 1.449752, 1.535108, 1.649820, 1.682369,
    1.519334, 1.519334, 1.486478, 1.413906, 1.335389, 1.374225, 1.392385, 1.424676, 1.529249, 1.705479, 1.967619, 2.046625,
    1.360759, 1.360759, 1.329878, 1.269765, 1.200485, 1.215329, 1.250150, 1.283948, 1.360961, 1.507495, 1.904621, 1.924856,
    1.175989, 1.175989, 1.168827, 1.121310, 1.083093, 1.104463, 1.128048, 1.169129, 1.242515, 1.355943, 1.741818, 1.833040,
    ])
