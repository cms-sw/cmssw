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
caloStage2Params.egCompressShapesLUTFile    = cms.FileInPath("L1Trigger/L1TCalorimeter/data/egCompressLUT_v4.txt")
caloStage2Params.egShapeIdType              = cms.string("compressed")
caloStage2Params.egShapeIdVersion           = cms.uint32(0)
caloStage2Params.egShapeIdLUTFile           = cms.FileInPath("L1Trigger/L1TCalorimeter/data/shapeIdentification_adapt0.99_compressedieta_compressedE_compressedshape_v15.12.08.txt")

caloStage2Params.egPUSType                  = cms.string("None")
caloStage2Params.egIsolationType            = cms.string("compressed")
#caloStage2Params.egIsoLUTFile               = cms.FileInPath("L1Trigger/L1TCalorimeter/data/IsoIdentification_adapt_extrap_v16.07.29.txt")
caloStage2Params.egIsoLUTFile               = cms.FileInPath("L1Trigger/L1TCalorimeter/data/IsoIdentification_adapt_extrap_v16.08.08.txt") # new SK Sep '16
caloStage2Params.egIsoAreaNrTowersEta       = cms.uint32(2)
caloStage2Params.egIsoAreaNrTowersPhi       = cms.uint32(4)
caloStage2Params.egIsoVetoNrTowersPhi       = cms.uint32(2)
#caloStage2Params.egIsoPUEstTowerGranularity = cms.uint32(1)
#caloStage2Params.egIsoMaxEtaAbsForTowerSum  = cms.uint32(4)
#caloStage2Params.egIsoMaxEtaAbsForIsoSum    = cms.uint32(27)
caloStage2Params.egPUSParams                = cms.vdouble(1,4,32) #Isolation window in firmware goes up to abs(ieta)=32 for now
caloStage2Params.egCalibrationType          = cms.string("compressed")
caloStage2Params.egCalibrationVersion       = cms.uint32(0)
caloStage2Params.egCalibrationLUTFile       = cms.FileInPath("L1Trigger/L1TCalorimeter/data/corrections_Trimming10_compressedieta_compressedE_compressedshape_v16.03.14.txt")


# Tau
caloStage2Params.tauLsb                        = cms.double(0.5)
caloStage2Params.tauSeedThreshold              = cms.double(0.)
caloStage2Params.tauNeighbourThreshold         = cms.double(0.)
caloStage2Params.tauIsoAreaNrTowersEta         = cms.uint32(2)
caloStage2Params.tauIsoAreaNrTowersPhi         = cms.uint32(4)
caloStage2Params.tauIsoVetoNrTowersPhi         = cms.uint32(2)
caloStage2Params.tauPUSType                    = cms.string("None")
caloStage2Params.tauIsoLUTFile                 = cms.FileInPath("L1Trigger/L1TCalorimeter/data/Tau_Iso_LUT_Option_22_NewLayer1Calibration_noCompressionBlock_SK1616_EmuOldFormat_v6.2.0.txt")
caloStage2Params.tauIsoLUTFile2                = cms.FileInPath("L1Trigger/L1TCalorimeter/data/Tau_Iso_LUT_Option_22_NewLayer1Calibration_noCompressionBlock_SK1616_EmuOldFormat_v6.2.0.txt")
caloStage2Params.tauCalibrationLUTFile         = cms.FileInPath("L1Trigger/L1TCalorimeter/data/Tau_Calibration_LUT_NewLayer1Calibration_SK1616_EmuOldFormat_v11.0.0.txt")
caloStage2Params.tauCompressLUTFile            = cms.FileInPath("L1Trigger/L1TCalorimeter/data/tauCompressAllLUT_12bit_v3.txt")
caloStage2Params.tauPUSParams                  = cms.vdouble(1,4,32)

# jets
caloStage2Params.jetLsb                = cms.double(0.5)
caloStage2Params.jetSeedThreshold      = cms.double(4.0)
caloStage2Params.jetNeighbourThreshold = cms.double(0.)
caloStage2Params.jetPUSType            = cms.string("ChunkyDonut")

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

caloStage2Params.jetCompressPtLUTFile     = cms.FileInPath("L1Trigger/L1TCalorimeter/data/lut_pt_compress.txt")
caloStage2Params.jetCompressEtaLUTFile    = cms.FileInPath("L1Trigger/L1TCalorimeter/data/lut_30to40_hfHighPt_experiment2_changeLimits_eta.txt")
caloStage2Params.jetCalibrationLUTFile    = cms.FileInPath("L1Trigger/L1TCalorimeter/data/lut_30to40_hfHighPt_experiment2_changeLimits_add_mult.txt")


# sums: 0=ET, 1=HT, 2=MET, 3=MHT
caloStage2Params.etSumLsb                = cms.double(0.5)
caloStage2Params.etSumEtaMin             = cms.vint32(1, 1, 1, 1, 1)
caloStage2Params.etSumEtaMax             = cms.vint32(28,  28,  28,  28, 27)
caloStage2Params.etSumEtThreshold        = cms.vdouble(0.,  30.,  0.,  30., 0.)

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
#New ECAL calibrations binned on tower ET 
caloStage2Params.layer1ECalScaleETBins = cms.vint32([10, 15, 20, 25, 30, 35, 40, 45, 256])
caloStage2Params.layer1ECalScaleFactors = cms.vdouble([
        1.208190, 1.193831, 1.193843, 1.214704, 1.192821, 1.220908, 1.218467, 1.223580, 1.217246, 1.237422, 1.242979, 1.251475, 1.270360, 1.316498, 1.304969, 1.313628, 1.412251, 2.141862, 1.416868, 1.528588, 1.537433, 1.530344, 
        1.513922, 1.553387, 1.567000, 1.541610, 1.457139, 1.487300, 1.127397, 1.140318, 1.153732, 1.177768, 1.152771, 1.155775, 1.160308, 1.158241, 1.169187, 1.174276, 1.170810, 1.188320, 1.200725, 1.209210, 1.213054, 1.235987, 
        1.312487, 1.937033, 1.270815, 1.359124, 1.394787, 1.368162, 1.364445, 1.395063, 1.434056, 1.420153, 1.346325, 1.402988, 1.089261, 1.084059, 1.090366, 1.089138, 1.090491, 1.103439, 1.092586, 1.096681, 1.112075, 1.115814, 
        1.114636, 1.118974, 1.139433, 1.152043, 1.155630, 1.157002, 1.205450, 1.854968, 1.225174, 1.299449, 1.353789, 1.316810, 1.314123, 1.349183, 1.395382, 1.381582, 1.317593, 1.384600, 1.071482, 1.067483, 1.067462, 1.066326, 
        1.071064, 1.077144, 1.073465, 1.072841, 1.086271, 1.104900, 1.081486, 1.086756, 1.096188, 1.114686, 1.122217, 1.129874, 1.168171, 1.812662, 1.202279, 1.279068, 1.326350, 1.295637, 1.290608, 1.324722, 1.369783, 1.361360, 
        1.299822, 1.372281, 1.061298, 1.057444, 1.055856, 1.055508, 1.056166, 1.074645, 1.066611, 1.061034, 1.065241, 1.084247, 1.077634, 1.076026, 1.082194, 1.097146, 1.109076, 1.121422, 1.142658, 1.769411, 1.183381, 1.252022, 
        1.315308, 1.265650, 1.268436, 1.305292, 1.349661, 1.337928, 1.285204, 1.357143, 1.050072, 1.047984, 1.048092, 1.045895, 1.049023, 1.070026, 1.054519, 1.058549, 1.056486, 1.078440, 1.063950, 1.065979, 1.077038, 1.091099, 
        1.080287, 1.101142, 1.130997, 1.661993, 1.181250, 1.230554, 1.275224, 1.256848, 1.264708, 1.297300, 1.342283, 1.332615, 1.279686, 1.355711, 1.048233, 1.049490, 1.046896, 1.044864, 1.044078, 1.057606, 1.049355, 1.054238, 
        1.055968, 1.070471, 1.059268, 1.057748, 1.068407, 1.096938, 1.077940, 1.084436, 1.123254, 1.699847, 1.164002, 1.231507, 1.277487, 1.251166, 1.249815, 1.280204, 1.329053, 1.321746, 1.273126, 1.349214, 1.038026, 1.032367, 
        1.035255, 1.039838, 1.042236, 1.062323, 1.047012, 1.048638, 1.048305, 1.068980, 1.050191, 1.051813, 1.064593, 1.083348, 1.064224, 1.076602, 1.115396, 1.625167, 1.165931, 1.218367, 1.249837, 1.242801, 1.241377, 1.276845, 
        1.319916, 1.315425, 1.269162, 1.345198, 1.117350, 1.117560, 1.117560, 1.118255, 1.119337, 1.125571, 1.121147, 1.121547, 1.123242, 1.128062, 1.123738, 1.125514, 1.128207, 1.135966, 1.132615, 1.136350, 1.155429, 1.319682, 
        1.190906, 1.211385, 1.204970, 1.224400, 1.225175, 1.246255, 1.277491, 1.270270, 1.277960, 1.309594,
    ])
caloStage2Params.layer1HCalScaleETBins = cms.vint32([10, 15, 20, 25, 30, 35, 40, 45, 256])
caloStage2Params.layer1HCalScaleFactors = cms.vdouble([
    0.938987, 1.305673, 1.477472, 0.765028, 1.815504, 1.798761, 0.778031, 1.236700, 1.095857, 1.378264, 1.433842, 1.167307, 1.352559, 1.690047, 0.891717, 1.854560, 1.793667, 1.553130, 1.707625, 1.916118, 1.481646, 1.541782, 1.490156, 1.479861, 1.468454, 1.495288, 1.610239, 1.588946,
    1.959768, 1.735311, 1.720083, 1.747394, 1.900315, 1.850038, 2.069131, 1.870382, 1.591709, 1.754415, 1.691840, 1.790095, 1.945210, 1.639001, 1.626256, 1.751506, 1.807836, 1.880716, 1.546263, 1.539627, 1.463240, 1.617592, 1.585718, 1.591975, 1.591791, 1.644489, 1.799823, 1.734512,
    1.797256, 1.578326, 1.692388, 1.503111, 1.672529, 1.519262, 1.582937, 1.886371, 1.778520, 1.612681, 1.398586, 1.603034, 1.454530, 1.328033, 1.471435, 1.636657, 1.527224, 1.512641, 1.410223, 1.388980, 1.421767, 1.572817, 1.578374, 1.588652, 1.621864, 1.729711, 1.918856, 1.803434,
    1.527994, 1.365275, 1.392104, 1.498491, 1.519979, 1.355727, 1.405808, 1.417970, 1.333893, 1.334817, 1.310798, 1.379249, 1.351143, 1.222402, 1.366848, 1.300051, 1.179617, 1.350694, 1.262175, 1.330276, 1.391770, 1.463739, 1.488979, 1.515451, 1.539866, 1.638972, 1.868056, 1.783022,
    1.258886, 1.373863, 1.306919, 1.356936, 1.370243, 1.385183, 1.251747, 1.269948, 1.232910, 1.298851, 1.233371, 1.269583, 1.283337, 1.207918, 1.230305, 1.229290, 1.279386, 1.274979, 1.274343, 1.273966, 1.336204, 1.425413, 1.427249, 1.446559, 1.494423, 1.585447, 1.732028, 1.837370,
    1.312747, 1.251113, 1.290323, 1.266130, 1.241814, 1.258689, 1.301229, 1.290575, 1.199071, 1.206718, 1.186692, 1.171748, 1.229947, 1.197701, 1.220179, 1.124275, 1.211136, 1.187607, 1.180801, 1.266214, 1.289450, 1.380122, 1.421375, 1.443506, 1.468912, 1.538188, 1.786963, 1.786963,
    1.261641, 1.242962, 1.289171, 1.176940, 1.187245, 1.230983, 1.142716, 1.183702, 1.166847, 1.166354, 1.134944, 1.191856, 1.156877, 1.177983, 1.108505, 1.149493, 1.141904, 1.158885, 1.151391, 1.183157, 1.277163, 1.366516, 1.373016, 1.392367, 1.425116, 1.527114, 1.527114, 1.527114,
    1.228298, 1.191839, 1.256205, 1.201946, 1.129291, 1.186304, 1.133396, 1.101866, 1.210155, 1.100159, 1.108861, 1.093585, 1.126787, 1.106975, 1.153106, 1.112337, 1.142352, 1.155770, 1.124314, 1.167838, 1.241431, 1.299161, 1.329571, 1.335993, 1.378090, 1.378090, 1.378090, 1.378090,
    1.115713, 1.101212, 1.093974, 1.094815, 1.090964, 1.072051, 1.062362, 1.035872, 1.031726, 1.023072, 1.010703, 1.015952, 1.020603, 1.024692, 1.025791, 1.156998, 1.096629, 1.110618, 1.091158, 1.122913, 1.195124, 1.271093, 1.277982, 1.321275, 1.321275, 1.321275, 1.321275, 1.321275,
    ])
# HF 1x1 scale factors will be a 5*12 array:
#  12 eta scale factors (30-41)
#  in 5 REAL ET bins (5, 20, 30, 50, Max)
#  So, index = etBin*12+ietaHF
caloStage2Params.layer1HFScaleETBins = cms.vint32([5, 20, 30, 50, 256])
caloStage2Params.layer1HFScaleFactors = cms.vdouble([
    1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 
    1.767080, 1.767080, 1.755186, 1.769951, 1.763527, 1.791043, 1.898787, 1.982235, 2.071074, 2.193011, 2.356886, 2.403384, 
    2.170477, 2.170477, 2.123540, 2.019866, 1.907698, 1.963179, 1.989122, 2.035251, 2.184642, 2.436399, 2.810884, 2.923750, 
    1.943941, 1.943941, 1.899826, 1.813950, 1.714978, 1.736184, 1.785928, 1.834211, 1.944230, 2.153565, 2.720887, 2.749795, 
    1.679984, 1.679984, 1.669753, 1.601871, 1.547276, 1.577805, 1.611497, 1.670184, 1.775022, 1.937061, 2.488311, 2.618629, 
    ])
