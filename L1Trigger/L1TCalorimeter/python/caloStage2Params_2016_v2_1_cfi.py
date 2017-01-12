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
caloStage2Params.egIsoLUTFile               = cms.FileInPath("L1Trigger/L1TCalorimeter/data/IsoIdentification_0.25_adapt_extrap_v16.04.05.txt")
caloStage2Params.egIsoAreaNrTowersEta       = cms.uint32(2)
caloStage2Params.egIsoAreaNrTowersPhi       = cms.uint32(4)
caloStage2Params.egIsoVetoNrTowersPhi       = cms.uint32(3)
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
caloStage2Params.tauIsoLUTFile                 = cms.FileInPath("L1Trigger/L1TCalorimeter/data/Tau_Iso_LUT_Option_5_Layer1Calibration_noCompressionBlock_v4.0.0.txt")
caloStage2Params.tauCalibrationLUTFile         = cms.FileInPath("L1Trigger/L1TCalorimeter/data/Tau_Calibration_LUT_Layer1Calibration_v9.0.0.txt")
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
caloStage2Params.jetCompressEtaLUTFile    = cms.FileInPath("L1Trigger/L1TCalorimeter/data/lut_eta_compress.txt")
caloStage2Params.jetCalibrationLUTFile    = cms.FileInPath("L1Trigger/L1TCalorimeter/data/lut_add_mult.txt")


# sums: 0=ET, 1=HT, 2=MET, 3=MHT
caloStage2Params.etSumLsb                = cms.double(0.5)
caloStage2Params.etSumEtaMin             = cms.vint32(1, 1, 1, 1)
caloStage2Params.etSumEtaMax             = cms.vint32(28,  28,  28,  28)
caloStage2Params.etSumEtThreshold        = cms.vdouble(0.,  30.,  0.,  30.)

caloStage2Params.etSumXPUSLUTFile         = cms.FileInPath("L1Trigger/L1TCalorimeter/data/lut_etSumPUS_dummy.txt")
caloStage2Params.etSumYPUSLUTFile         = cms.FileInPath("L1Trigger/L1TCalorimeter/data/lut_etSumPUS_dummy.txt")
caloStage2Params.etSumEttPUSLUTFile       = cms.FileInPath("L1Trigger/L1TCalorimeter/data/lut_etSumPUS_dummy.txt")
caloStage2Params.etSumEcalSumPUSLUTFile   = cms.FileInPath("L1Trigger/L1TCalorimeter/data/lut_etSumPUS_dummy.txt")


# Layer 1 LUT specification
#
# Et-dependent scale factors
# ECal/HCal scale factors will be a 9*28 array:
#   28 eta scale factors (1-28)
#   in 9 ET bins (10, 15, 20, 25, 30, 35, 40, 45, Max)
#  So, index = etBin*28+ieta
caloStage2Params.layer1ECalScaleETBins = cms.vint32([10, 15, 20, 25, 30, 35, 40, 45, 256])
caloStage2Params.layer1ECalScaleFactors = cms.vdouble([
    1.1847, 1.16759, 1.17779, 1.19955, 1.21125, 1.214, 1.21503, 1.22515, 1.24151, 1.27836, 1.30292, 1.33526, 1.42338, 1.4931, 1.49597, 1.50405, 1.52785, 1.81552, 1.59856, 1.75692, 1.76496, 1.77562, 1.69527, 1.66827, 1.61861, 1.56645, 1, 1,
    1.1351, 1.12589, 1.12834, 1.13725, 1.14408, 1.1494, 1.14296, 1.14852, 1.1578, 1.17634, 1.18038, 1.19386, 1.23758, 1.27605, 1.27818, 1.28195, 1.34881, 1.71053, 1.37338, 1.52571, 1.54801, 1.53316, 1.4397, 1.40497, 1.37743, 1.33914, 1, 1,
    1.18043, 1.17823, 1.1751, 1.17608, 1.19152, 1.196, 1.20125, 1.2068, 1.22584, 1.22476, 1.22395, 1.22302, 1.25137, 1.28097, 1.29871, 1.2862, 1.33489, 1.60937, 1.28365, 1.41367, 1.42521, 1.42041, 1.36784, 1.34922, 1.32754, 1.29825, 1, 1,
    1.11664, 1.11852, 1.11861, 1.12367, 1.12405, 1.14814, 1.14304, 1.15337, 1.16607, 1.18698, 1.17048, 1.17463, 1.2185, 1.23842, 1.23214, 1.24744, 1.30047, 1.47152, 1.22868, 1.33121, 1.34841, 1.35178, 1.30048, 1.28537, 1.27012, 1.24159, 1, 1,
    1.08422, 1.08146, 1.08706, 1.08906, 1.08636, 1.10092, 1.10363, 1.11102, 1.1186, 1.13301, 1.12369, 1.14377, 1.16477, 1.17801, 1.18782, 1.17168, 1.24593, 1.36835, 1.20252, 1.28349, 1.29828, 1.30328, 1.26848, 1.25817, 1.2464, 1.22259, 1, 1,
    1.07444, 1.06774, 1.06883, 1.0707, 1.07881, 1.08859, 1.08285, 1.08747, 1.09736, 1.10678, 1.10008, 1.10717, 1.12858, 1.15383, 1.15826, 1.14855, 1.19911, 1.32567, 1.17553, 1.25976, 1.27926, 1.28459, 1.24524, 1.23706, 1.22597, 1.20006, 1, 1,
    1.06224, 1.05968, 1.05767, 1.06254, 1.06729, 1.0691, 1.07125, 1.07312, 1.08124, 1.08966, 1.08695, 1.08826, 1.10611, 1.13115, 1.12641, 1.13093, 1.17074, 1.28958, 1.16217, 1.22844, 1.24812, 1.25352, 1.22065, 1.21287, 1.20544, 1.18344, 1, 1,
    1.03589, 1.03224, 1.03229, 1.03623, 1.03979, 1.04403, 1.04574, 1.049, 1.04821, 1.06183, 1.0588, 1.06655, 1.08582, 1.10289, 1.10052, 1.10506, 1.143, 1.27373, 1.1459, 1.2156, 1.23455, 1.23968, 1.20753, 1.20127, 1.19629, 1.16809, 1, 1,
    1.03456, 1.02955, 1.03079, 1.03509, 1.03949, 1.0437, 1.04236, 1.04486, 1.0517, 1.05864, 1.05516, 1.06167, 1.07738, 1.0985, 1.09317, 1.09559, 1.13557, 1.26076, 1.14118, 1.20545, 1.22137, 1.22802, 1.19936, 1.19676, 1.19088, 1.16709, 1, 1,
    ])
caloStage2Params.layer1HCalScaleETBins = cms.vint32([10, 15, 20, 25, 30, 35, 40, 45, 256])
caloStage2Params.layer1HCalScaleFactors = cms.vdouble([
    1.511112, 1.519900, 1.499483, 1.488560, 1.528111, 1.475114, 1.476616, 1.514163, 1.515306, 1.542464, 1.511663, 1.593745, 1.493667, 1.485315, 1.419925, 1.349169, 1.312518, 1.423302, 1.478461, 1.525868, 1, 1, 1, 1, 1, 1, 1, 1,
    1.383350, 1.365700, 1.368470, 1.354610, 1.348480, 1.329720, 1.272250, 1.301710, 1.322210, 1.360860, 1.333850, 1.392200, 1.403060, 1.394870, 1.322050, 1.244570, 1.206910, 1.321870, 1.344160, 1.403270, 1, 1, 1, 1, 1, 1, 1, 1,
    1.245690, 1.238320, 1.245420, 1.234830, 1.243730, 1.249790, 1.179450, 1.213620, 1.219030, 1.252130, 1.209560, 1.250710, 1.280490, 1.262800, 1.254060, 1.186810, 1.127830, 1.260000, 1.275140, 1.305850, 1, 1, 1, 1, 1, 1, 1, 1,
    1.189940, 1.189120, 1.177120, 1.179690, 1.185510, 1.150590, 1.151830, 1.167860, 1.154310, 1.163190, 1.161700, 1.136100, 1.161870, 1.195050, 1.153910, 1.117900, 1.106750, 1.208120, 1.160020, 1.232800, 1, 1, 1, 1, 1, 1, 1, 1,
    1.122540, 1.129520, 1.125080, 1.115150, 1.118250, 1.096190, 1.108170, 1.087490, 1.109750, 1.099780, 1.081000, 1.050610, 1.078270, 1.079460, 1.047740, 1.041400, 1.041750, 1.116880, 1.097730, 1.125780, 1, 1, 1, 1, 1, 1, 1, 1,
    1.110470, 1.117340, 1.115980, 1.088490, 1.088260, 1.078230, 1.062720, 1.054690, 1.053270, 1.086640, 1.050620, 1.038470, 1.046440, 1.059130, 1.012240, 1.039030, 1.036040, 1.088460, 1.078880, 1.090600, 1, 1, 1, 1, 1, 1, 1, 1,
    1.115970, 1.111010, 1.113170, 1.079390, 1.076850, 1.063730, 1.039300, 1.049910, 1.040100, 1.025820, 1.015830, 1.015850, 1.010810, 1.014210, 0.980321, 1.023580, 1.045990, 1.073220, 1.057750, 1.059850, 1, 1, 1, 1, 1, 1, 1, 1,
    1.061180, 1.059770, 1.071210, 1.064420, 1.065340, 1.043070, 1.041400, 1.022680, 1.017410, 1.017690, 1.005610, 1.006360, 0.999420, 0.990866, 0.986723, 0.989036, 0.995116, 1.045620, 1.024330, 1.040660, 1, 1, 1, 1, 1, 1, 1, 1,
    1.083150, 1.067090, 1.083180, 1.061010, 1.075640, 1.051640, 1.038760, 1.042670, 1.010910, 1.011580, 1.006560, 0.984468, 0.986642, 0.985799, 0.968133, 1.000290, 1.011210, 1.046690, 1.016670, 1.020470, 1, 1, 1, 1, 1, 1, 1, 1,
    ])
# HF 1x1 scale factors will be a 5*12 array:
#  12 eta scale factors (30-41)
#  in 5 REAL ET bins (5, 20, 30, 50, Max)
#  So, index = etBin*12+ietaHF
caloStage2Params.layer1HFScaleETBins = cms.vint32([5, 20, 30, 50, 256])
caloStage2Params.layer1HFScaleFactors = cms.vdouble([
    1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 
    1.55, 1.49, 1.35, 1.29, 1.30, 1.42, 1.49, 1.59, 1.74, 1.86, 2.18, 2.43, 
    1.60, 1.51, 1.38, 1.38, 1.44, 1.56, 1.60, 1.67, 1.83, 2.02, 2.66, 2.79, 
    1.56, 1.41, 1.34, 1.42, 1.44, 1.52, 1.57, 1.63, 1.73, 1.94, 2.64, 2.64, 
    1.46, 1.30, 1.29, 1.43, 1.42, 1.49, 1.52, 1.59, 1.69, 1.87, 2.49, 2.66, 
    ])
