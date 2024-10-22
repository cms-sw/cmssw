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
caloStage2Params.jetCalibrationLUTFile    = cms.FileInPath("L1Trigger/L1TCalorimeter/data/lut_calib_2017v3.txt")


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
# ECAL and HCAl calibrations using mode
caloStage2Params.layer1ECalScaleETBins = cms.vint32([6, 9, 12, 15, 20, 25, 30, 35, 40, 45, 55, 70, 256])
caloStage2Params.layer1ECalScaleFactors = cms.vdouble([
        1.121683, 1.123691, 1.108489, 1.121521, 1.125255, 1.121693, 1.127835, 1.117779, 1.127152, 1.131185, 1.124344, 1.128536, 1.134725, 1.141772, 1.143249, 1.140299, 1.152531, 1.319443, 
        1.210847, 1.339125, 1.356573, 1.396528, 1.377966, 1.409579, 1.443289, 1.454007, 1.295642, 1.227788, 1.082900, 1.074461, 1.079962, 1.075127, 1.086775, 1.083590, 1.084558, 1.082557, 
        1.090276, 1.087614, 1.091867, 1.087652, 1.099034, 1.103327, 1.100456, 1.109527, 1.110784, 1.302258, 1.168732, 1.278368, 1.298565, 1.331746, 1.314623, 1.341301, 1.386985, 1.430218, 
        1.204316, 1.190891, 1.062883, 1.054139, 1.058559, 1.063505, 1.063059, 1.062662, 1.065150, 1.068217, 1.065320, 1.067348, 1.067776, 1.070561, 1.077374, 1.085696, 1.082619, 1.079525, 
        1.094581, 1.248178, 1.148441, 1.236617, 1.267048, 1.288578, 1.288512, 1.315666, 1.360049, 1.362456, 1.170797, 1.167626, 1.050682, 1.041363, 1.049212, 1.049985, 1.050699, 1.049319, 
        1.051826, 1.051975, 1.057911, 1.055179, 1.053420, 1.054771, 1.069557, 1.073795, 1.070160, 1.069165, 1.072477, 1.238395, 1.138875, 1.216007, 1.244247, 1.273514, 1.271499, 1.294721, 
        1.338348, 1.387534, 1.151556, 1.158395, 1.042771, 1.034424, 1.035937, 1.035207, 1.040111, 1.040705, 1.041693, 1.042686, 1.039497, 1.043452, 1.041545, 1.044947, 1.053061, 1.062762, 
        1.058935, 1.055335, 1.065606, 1.193257, 1.131426, 1.198880, 1.224663, 1.251683, 1.256435, 1.269958, 1.318988, 1.372872, 1.136677, 1.141251, 1.033837, 1.031071, 1.029707, 1.031193, 
        1.033072, 1.035870, 1.031507, 1.035295, 1.034799, 1.035418, 1.038483, 1.042567, 1.044793, 1.053306, 1.052018, 1.051434, 1.056204, 1.165940, 1.122919, 1.194796, 1.215185, 1.241700, 
        1.241192, 1.256956, 1.299124, 1.357881, 1.123170, 1.134848, 1.030442, 1.024761, 1.025480, 1.025364, 1.026802, 1.028792, 1.026702, 1.029903, 1.030029, 1.031292, 1.031683, 1.032099, 
        1.038992, 1.046039, 1.044969, 1.042792, 1.049454, 1.160109, 1.119586, 1.178622, 1.205947, 1.229264, 1.231618, 1.245986, 1.291842, 1.358345, 1.117718, 1.126883, 1.025564, 1.020994, 
        1.022154, 1.022523, 1.024304, 1.024387, 1.024369, 1.024698, 1.024857, 1.026679, 1.026999, 1.027745, 1.036174, 1.040190, 1.041848, 1.040525, 1.043341, 1.148116, 1.118444, 1.173695, 
        1.200880, 1.225635, 1.219548, 1.240240, 1.278999, 1.315860, 1.112813, 1.122397, 1.023703, 1.017579, 1.018104, 1.017659, 1.020570, 1.021379, 1.021771, 1.022549, 1.022039, 1.020696, 
        1.023734, 1.024919, 1.033031, 1.039501, 1.037120, 1.038029, 1.044373, 1.147821, 1.115866, 1.171247, 1.195742, 1.219918, 1.222375, 1.232918, 1.275971, 1.312366, 1.104763, 1.123853, 
        1.022729, 1.015033, 1.016055, 1.024841, 1.018966, 1.019355, 1.019383, 1.022111, 1.019620, 1.018730, 1.023090, 1.023060, 1.031683, 1.037654, 1.033611, 1.035408, 1.042514, 1.129705, 
        1.111494, 1.170215, 1.192696, 1.211153, 1.219209, 1.226238, 1.268898, 1.329354, 1.099919, 1.119065, 1.019639, 1.012224, 1.019498, 1.014690, 1.015372, 1.015029, 1.016340, 1.018439, 
        1.017241, 1.015631, 1.019250, 1.020679, 1.029578, 1.033115, 1.032704, 1.032550, 1.038186, 1.130066, 1.122241, 1.163237, 1.187682, 1.205797, 1.213205, 1.220996, 1.257875, 1.319851, 
        1.095214, 1.114397, 1.014848, 1.007609, 1.014595, 1.009661, 1.017298, 1.011264, 1.012152, 1.014643, 1.013504, 1.009988, 1.016762, 1.018233, 1.025658, 1.029184, 1.031794, 1.031796, 
        1.035437, 1.119702, 1.119120, 1.154691, 1.181214, 1.192774, 1.189330, 1.214413, 1.253684, 1.313831, 1.099976, 1.048771, 1.018860, 1.002584, 1.024954, 1.005857, 1.013041, 1.018629, 
        1.008655, 1.009616, 1.010172, 1.007577, 1.013120, 1.015258, 1.022823, 1.028055, 1.029566, 1.029339, 1.033359, 1.148478, 1.112274, 1.150181, 1.171471, 1.184794, 1.188237, 1.212775, 
        1.244559, 1.259219, 1.099976, 1.048771
        
    ])
caloStage2Params.layer1HCalScaleETBins = cms.vint32([6, 9, 12, 15, 20, 25, 30, 35, 40, 45, 55, 70, 256])
caloStage2Params.layer1HCalScaleFactors = cms.vdouble([
        1.306665, 1.298686, 1.299093, 1.304538, 1.309732, 1.318956, 1.316933, 1.318530, 1.321018, 1.337300, 1.327375, 1.335612, 1.344994, 1.351881, 1.341971, 1.346799, 
        1.342065, 1.226200, 1.233410, 1.249447, 1.236383, 1.246362, 1.244474, 1.248714, 1.253319, 1.248255, 1.332186, 1.309733, 1.281192, 1.275692, 1.276582, 1.281867, 1.285152, 1.297746, 
        1.296706, 1.296025, 1.299368, 1.313771, 1.309365, 1.314090, 1.319631, 1.329637, 1.321478, 1.316573, 1.305902, 1.180456, 1.205533, 1.222687, 1.200736, 1.205808, 1.206095, 1.210354, 
        1.211630, 1.213855, 1.291656, 1.286007, 1.258863, 1.255918, 1.256607, 1.260180, 1.262566, 1.278135, 1.275185, 1.273841, 1.276996, 1.292344, 1.283939, 1.293121, 1.301536, 1.310902, 
        1.299557, 1.299160, 1.281618, 1.151744, 1.183925, 1.200305, 1.162737, 1.168838, 1.172386, 1.207352, 1.179348, 1.175249, 1.251152, 1.258246, 1.239417, 1.233119, 1.237800, 1.243426, 
        1.244901, 1.256973, 1.255848, 1.254187, 1.256834, 1.271202, 1.270373, 1.273155, 1.279182, 1.286435, 1.275181, 1.275483, 1.257610, 1.132167, 1.163909, 1.179439, 1.135584, 1.143493, 
        1.139635, 1.162142, 1.146630, 1.140251, 1.210845, 1.231968, 1.213200, 1.211401, 1.211789, 1.220689, 1.221046, 1.236659, 1.237336, 1.227459, 1.235608, 1.244857, 1.243908, 1.247274, 
        1.254118, 1.261175, 1.251198, 1.245863, 1.226670, 1.108465, 1.139574, 1.154562, 1.097707, 1.113260, 1.113130, 1.111061, 1.103470, 1.125493, 1.176002, 1.202533, 1.185774, 1.182431, 
        1.187055, 1.192933, 1.190270, 1.200067, 1.200865, 1.201043, 1.197957, 1.216726, 1.215289, 1.219775, 1.244460, 1.230195, 1.220220, 1.214304, 1.187170, 1.081579, 1.115047, 1.126990, 
        1.062215, 1.073020, 1.072449, 1.072747, 1.093658, 1.065245, 1.137046, 1.160033, 1.166813, 1.159113, 1.158696, 1.168500, 1.176224, 1.177458, 1.176049, 1.177527, 1.194460, 1.180978, 
        1.180605, 1.182946, 1.190139, 1.191530, 1.193145, 1.185069, 1.150335, 1.059742, 1.091350, 1.101493, 1.040157, 1.045679, 1.036077, 1.043668, 1.044040, 1.033542, 1.100558, 1.134767, 
        1.138914, 1.138707, 1.137686, 1.138994, 1.137928, 1.141699, 1.185898, 1.154198, 1.138946, 1.158737, 1.151058, 1.153530, 1.160459, 1.166764, 1.156826, 1.150647, 1.124801, 1.038918, 
        1.067086, 1.084463, 1.007674, 1.017804, 1.014871, 1.014488, 1.017093, 1.033303, 1.068936, 1.090506, 1.121376, 1.113413, 1.119699, 1.120693, 1.119095, 1.131076, 1.122907, 1.133613, 
        1.114476, 1.142076, 1.104291, 1.132805, 1.133099, 1.142017, 1.146185, 1.126269, 1.091932, 1.016653, 1.086545, 1.060298, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 
        1.038077, 1.074598, 1.103010, 1.095256, 1.102633, 1.143426, 1.102720, 1.104560, 1.109598, 1.109057, 1.107692, 1.111018, 1.100855, 1.112122, 1.111549, 1.106951, 1.117613, 1.107565, 
        1.074493, 1.003659, 1.036376, 1.046378, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.018726, 1.058204, 1.076980, 1.073889, 1.076191, 1.077141, 1.075920, 1.091658, 
        1.091658, 1.079356, 1.090271, 1.079933, 1.079696, 1.080747, 1.091206, 1.082266, 1.081607, 1.069378, 1.041722, 1.000000, 1.016669, 1.018956, 1.000000, 1.000000, 1.000000, 1.000000, 
        1.000000, 1.000000, 1.000000, 1.028275, 1.040284, 1.040510, 1.042076, 1.045636, 1.045923, 1.050276, 1.050721, 1.077816, 1.047166, 1.051137, 1.050572, 1.061542, 1.043422, 1.044104, 
        1.046123, 1.025588, 1.003596, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000526, 1.000110, 1.001580, 1.003231, 
        1.003303, 1.007173, 1.006385, 1.004558, 1.002023, 1.002172, 1.004252, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 
        1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000
    ])
# HF 1x1 scale factors will be a 13*12 array:
#  12 eta scale factors (30-41)
#  in 13 ET bins (6, 9, 12, 15, 20, 25, 30, 35, 40, 45, 55, 70, Max)
#  So, index = etBin*12+ietaHF
# HF energies were formerly multiplied by 0.7; this has been removed
caloStage2Params.layer1HFScaleETBins = cms.vint32([6, 9, 12, 15, 20, 25, 30, 35, 40, 45, 55, 70, 256])
caloStage2Params.layer1HFScaleFactors = cms.vdouble([
    1.436933, 1.150784, 1.198317, 1.185131, 1.203083, 1.244493, 1.276614, 1.305569, 1.343967, 1.456317, 1.801191, 1.922477,
    1.287897, 1.082326, 1.127774, 1.131999, 1.122464, 1.155969, 1.172212, 1.195422, 1.268606, 1.298257, 1.663198, 1.778371,
    1.212180, 1.057017, 1.115133, 1.087006, 1.054219, 1.086285, 1.096476, 1.111732, 1.175957, 1.218147, 1.592375, 1.704887,
    1.159324, 1.036555, 1.091943, 1.058415, 1.011008, 1.026250, 1.027465, 1.055655, 1.103695, 1.150204, 1.551786, 1.652949,
    1.098260, 1.016998, 1.059197, 1.021185, 1.011008, 1.026250, 1.027465, 1.007086, 1.045239, 1.097873, 1.504194, 1.612279,
    1.060359, 1.016998, 1.024919, 1.021185, 1.011008, 1.026250, 1.027465, 1.007086, 1.010229, 1.060789, 1.448109, 1.552224,
    1.028576, 1.016998, 1.009009, 1.021185, 1.011008, 1.026250, 1.027465, 1.007086, 1.010229, 1.036400, 1.401057, 1.504891,
    1.002432, 1.016998, 1.009009, 1.021185, 1.011008, 1.026250, 1.027465, 1.007086, 1.010229, 1.016859, 1.353666, 1.475716,
    1.002432, 1.016998, 1.009009, 1.021185, 1.011008, 1.026250, 1.027465, 1.007086, 1.010229, 1.002650, 1.296711, 1.444925,
    1.002432, 1.016998, 1.009009, 1.021185, 1.011008, 1.026250, 1.027465, 1.007086, 1.010229, 1.002650, 1.258159, 1.431994,
    1.002432, 1.016998, 1.009009, 1.021185, 1.011008, 1.026250, 1.027465, 1.007086, 1.010229, 1.002650, 1.203263, 1.411718,
    1.002432, 1.016998, 1.009009, 1.021185, 1.011008, 1.026250, 1.027465, 1.007086, 1.010229, 1.002650, 1.144271, 1.357477,
    1.002432, 1.016998, 1.009009, 1.021185, 1.011008, 1.026250, 1.027465, 1.007086, 1.010229, 1.002650, 1.098340, 1.283960
    ])
