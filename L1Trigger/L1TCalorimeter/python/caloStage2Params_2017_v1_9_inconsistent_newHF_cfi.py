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
        1.130334, 1.119962, 1.126024, 1.120406, 1.111179, 1.133527, 1.125995, 1.129625, 1.125964, 1.125258, 1.123375, 1.122048, 1.133815, 1.147838, 1.145630, 1.139293, 
        1.157534, 1.339242, 1.211603, 1.336350, 1.369878, 1.396782, 1.380363, 1.393728, 1.440300, 1.446334, 1.286519, 1.238250, 1.077088, 1.079060, 1.079096, 1.083203, 1.080727, 1.086077, 
        1.080533, 1.084457, 1.078272, 1.081497, 1.089963, 1.087940, 1.096228, 1.113445, 1.102903, 1.096925, 1.116196, 1.312593, 1.173224, 1.281942, 1.300656, 1.326027, 1.314928, 1.330207, 
        1.383218, 1.419373, 1.204300, 1.197184, 1.067812, 1.062978, 1.060785, 1.057179, 1.063547, 1.063266, 1.064877, 1.066625, 1.066378, 1.066932, 1.075625, 1.066453, 1.079413, 1.085061, 
        1.085168, 1.078691, 1.089709, 1.253270, 1.148869, 1.238069, 1.262793, 1.284366, 1.280102, 1.311317, 1.355314, 1.434084, 1.171654, 1.173208, 1.046492, 1.050406, 1.050959, 1.047304, 
        1.051662, 1.053587, 1.053855, 1.055744, 1.048710, 1.056439, 1.054754, 1.057336, 1.063035, 1.076792, 1.069485, 1.067776, 1.079069, 1.218793, 1.137883, 1.216621, 1.241998, 1.270041, 
        1.267882, 1.286495, 1.335127, 1.395486, 1.152940, 1.156142, 1.043978, 1.040136, 1.039025, 1.039330, 1.041385, 1.042915, 1.041122, 1.044398, 1.043794, 1.043640, 1.045099, 1.047485, 
        1.057906, 1.061544, 1.056825, 1.058442, 1.065407, 1.205149, 1.133437, 1.204112, 1.224333, 1.257891, 1.248124, 1.268428, 1.313134, 1.368784, 1.141208, 1.144097, 1.033457, 1.029218, 
        1.030435, 1.030538, 1.033459, 1.032981, 1.036532, 1.036703, 1.034598, 1.038112, 1.041808, 1.038690, 1.047063, 1.052805, 1.052508, 1.049937, 1.056335, 1.173834, 1.123604, 1.187141, 
        1.208633, 1.238479, 1.237508, 1.258150, 1.303871, 1.353362, 1.126547, 1.134980, 1.029715, 1.024030, 1.024750, 1.026206, 1.027932, 1.029315, 1.032252, 1.031480, 1.029974, 1.030765, 
        1.032377, 1.034623, 1.039962, 1.048350, 1.047832, 1.044557, 1.051589, 1.153050, 1.125013, 1.177329, 1.212756, 1.230148, 1.224597, 1.252599, 1.290018, 1.355427, 1.118939, 1.129079, 
        1.025536, 1.021728, 1.022106, 1.023307, 1.025435, 1.026387, 1.027254, 1.027180, 1.026101, 1.026589, 1.029924, 1.032588, 1.038345, 1.046162, 1.044055, 1.046443, 1.046608, 1.160459, 
        1.116849, 1.175012, 1.202504, 1.223553, 1.224158, 1.240761, 1.280735, 1.339369, 1.111984, 1.123868, 1.022273, 1.016136, 1.018392, 1.019256, 1.024367, 1.023904, 1.025006, 1.023479, 
        1.023952, 1.024099, 1.025444, 1.028970, 1.034119, 1.038280, 1.039033, 1.042159, 1.042588, 1.148945, 1.117443, 1.176734, 1.198446, 1.221529, 1.211286, 1.234116, 1.273238, 1.333888, 
        1.105810, 1.123098, 1.021078, 1.013334, 1.018946, 1.016669, 1.020571, 1.021188, 1.021485, 1.022544, 1.022017, 1.022114, 1.024772, 1.024332, 1.033237, 1.037666, 1.040766, 1.038437, 
        1.042256, 1.130386, 1.111927, 1.165524, 1.193367, 1.210252, 1.205264, 1.231160, 1.264775, 1.309169, 1.099833, 1.123041, 1.018228, 1.012285, 1.013980, 1.020144, 1.016896, 1.018384, 
        1.017817, 1.019727, 1.019084, 1.019362, 1.022384, 1.023937, 1.030328, 1.037102, 1.034043, 1.036503, 1.037696, 1.121815, 1.121559, 1.164390, 1.185663, 1.204823, 1.197527, 1.223703, 
        1.258725, 1.324007, 1.095566, 1.115319, 1.013546, 1.007820, 1.009638, 1.015188, 1.010788, 1.012038, 1.018315, 1.015675, 1.014850, 1.013848, 1.019112, 1.020198, 1.037507, 1.031925, 
        1.032069, 1.032889, 1.035325, 1.126545, 1.114475, 1.156333, 1.178635, 1.190262, 1.190437, 1.222668, 1.252851, 1.308670, 1.050266, 1.048401, 1.011587, 1.007820, 1.009551, 1.013411, 
        1.007016, 1.009525, 1.014389, 1.015981, 1.016168, 1.011305, 1.020560, 1.022187, 1.027350, 1.029644, 1.031880, 1.032638, 1.032912, 1.153441, 1.108623, 1.150289 , 1.170026, 1.188810, 
        1.178877, 1.201929, 1.247786, 1.293275, 1.050266, 1.048401,
        
    ])
caloStage2Params.layer1HCalScaleETBins = cms.vint32([6, 9, 12, 15, 20, 25, 30, 35, 40, 45, 55, 70, 256])
caloStage2Params.layer1HCalScaleFactors = cms.vdouble([
        1.312429, 1.302657, 1.300988, 1.302032, 1.307133, 1.321599, 1.318192, 1.317203, 1.323222, 1.338748, 1.328551, 1.334501, 1.346351, 1.349347, 1.342995, 1.343628, 1.339676, 1.222754, 
        1.230436, 1.245418, 1.231993, 1.240127, 1.242939, 1.244029, 1.247786, 1.242274, 1.329409, 1.306735, 1.285443, 1.277333, 1.279259, 1.282926, 1.285633, 1.299634, 1.297552, 1.295508, 
        1.302257, 1.314299, 1.308241, 1.314227, 1.320904, 1.326978, 1.320350, 1.320275, 1.307392, 1.173643, 1.201628, 1.218395, 1.195738, 1.200045, 1.205938, 1.205873, 1.207201, 1.209854, 
        1.290540, 1.283796, 1.259032, 1.253267, 1.256193, 1.261758, 1.263389, 1.277413, 1.275080, 1.273929, 1.279380, 1.294686, 1.287925, 1.293438, 1.301842, 1.310697, 1.300390, 1.297630,
        1.283912, 1.149293, 1.182525, 1.197255, 1.159113, 1.164678, 1.171197, 1.174489, 1.178128, 1.172870, 1.250786, 1.257251, 1.239805, 1.231959, 1.238541, 1.238098, 1.245785, 1.257203, 
        1.254622, 1.256197, 1.261608, 1.275195, 1.274428, 1.274079, 1.283237, 1.289679, 1.278452, 1.274505, 1.260665, 1.131537, 1.159406, 1.176042, 1.132153, 1.142463, 1.139941, 1.147995, 
        1.138997, 1.139102, 1.210676, 1.230570, 1.212268, 1.209235, 1.210689, 1.218892, 1.220183, 1.235480, 1.235570, 1.234393, 1.233797, 1.247858, 1.247209, 1.249051, 1.264595, 1.257831, 
        1.257500, 1.252682, 1.230194, 1.107754, 1.136731, 1.151359, 1.102662, 1.112685, 1.106814, 1.095626, 1.105728, 1.104554, 1.176118, 1.197959, 1.186273, 1.180414, 1.186408, 1.188159, 
        1.191924, 1.198158, 1.198725, 1.197503, 1.205352, 1.214207, 1.213476, 1.216046, 1.218184, 1.226513, 1.217714, 1.208788, 1.184840, 1.080976, 1.114328, 1.126873, 1.066873, 1.073007, 
        1.072881, 1.072953, 1.059609, 1.074772, 1.127481, 1.160618, 1.159078, 1.157182, 1.164886, 1.154836, 1.157914, 1.176514, 1.174068, 1.176280, 1.197439, 1.179142, 1.183980, 1.185142, 
        1.185735, 1.196947, 1.189154, 1.193340, 1.104360, 1.056341, 1.091648, 1.102191, 1.025580, 1.045614, 1.060314, 1.045408, 1.045643, 1.043473, 1.099534, 1.139393, 1.138604, 1.130180,
        1.136523, 1.137843, 1.136914, 1.149154, 1.151382, 1.153098, 1.122844, 1.157851, 1.149439, 1.150974, 1.151026, 1.161806, 1.152825, 1.145824, 1.123225, 1.037807, 1.089805, 1.084553, 
        1.007716, 1.014838, 1.017549, 1.024829, 1.026830, 1.016476, 1.066288, 1.089889, 1.120506, 1.110645, 1.118902, 1.118805, 1.118831, 1.122219, 1.129816, 1.133631, 1.148116, 1.131240, 
        1.126342, 1.120993, 1.131791, 1.133455, 1.135907, 1.165640, 1.088229, 1.018096, 1.049134, 1.060228, 1.007716, 1.014838, 1.005670, 1.024829, 1.026830, 1.002294, 1.046901, 1.073104,
        1.095492, 1.093788, 1.102188, 1.102476, 1.101457, 1.104054, 1.103630, 1.108289, 1.098593, 1.120296, 1.099002, 1.114210, 1.109549, 1.103429, 1.115524, 1.099283, 1.073944, 1.004872, 
        1.036008, 1.045825, 1.007716, 1.014838, 1.005670, 1.024829, 1.026830, 1.002294, 1.032847, 1.058203, 1.075976, 1.069039, 1.075807, 1.075813, 1.084858, 1.083828, 1.091137, 1.084421,
        1.085235, 1.091557, 1.077747, 1.078301, 1.113494, 1.079615, 1.083187, 1.065595, 1.030002, 1.004872, 1.018090, 1.018703, 1.007716, 1.014838, 1.005670, 1.024829, 1.026830, 1.002294, 
        1.032847, 1.028675, 1.039919, 1.037851, 1.041172, 1.043106, 1.043830, 1.049700, 1.054381, 1.048301, 1.046833, 1.048423, 1.046860, 1.019303, 1.042652, 1.032596, 1.044473, 1.024725, 
        1.002399, 1.004872, 1.018090, 1.005287, 1.007716, 1.014838, 1.005670, 1.024829, 1.026830, 1.040560, 1.032847, 1.028675, 1.000670, 1.037851, 1.000402, 1.000753, 1.024783, 1.006964, 
        1.006178, 1.012921, 1.002187, 1.001676, 1.046860, 1.019303, 1.042652, 1.032596, 1.044473, 1.056054, 1.037224, 1.004872, 1.018090, 1.005287, 1.007716, 1.014838, 1.005670, 1.024829, 
        1.026830, 1.040560, 1.032847, 1.028675
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
#caloStage2Params.layer1HFScaleFactors = cms.vdouble([
#    1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
#    1.236956, 1.236956, 1.228630, 1.238966, 1.234469, 1.253730, 1.329151, 1.387564, 1.449752, 1.535108, 1.649820, 1.682369,
#    1.519334, 1.519334, 1.486478, 1.413906, 1.335389, 1.374225, 1.392385, 1.424676, 1.529249, 1.705479, 1.967619, 2.046625,
#    1.360759, 1.360759, 1.329878, 1.269765, 1.200485, 1.215329, 1.250150, 1.283948, 1.360961, 1.507495, 1.904621, 1.924856,
#    1.175989, 1.175989, 1.168827, 1.121310, 1.083093, 1.104463, 1.128048, 1.169129, 1.242515, 1.355943, 1.741818, 1.833040,
#    ])



# HF 1x1 scale factors will be a 13*12 array:
#  12 eta scale factors (30-41)
#  in 13 ET bins (6, 9, 12, 15, 20, 25, 30, 35, 40, 45, 55, 70, Max)
#  So, index = etBin*12+ietaHF
# HF energies were formerly multiplied by 0.7; this has been removed
caloStage2Params.layer1HFScaleETBins = cms.vint32([6, 9, 12, 15, 20, 25, 30, 35, 40, 45, 55, 70, 256])
caloStage2Params.layer1HFScaleFactors = cms.vdouble([
    1.425608, 1.144793, 1.191335, 1.181760, 1.193822, 1.230082, 1.272377, 1.299459, 1.341597, 1.430573, 1.786881, 1.928974, 
    1.269741, 1.077939, 1.128736, 1.119878, 1.110004, 1.138991, 1.167746, 1.199128, 1.261051, 1.291146, 1.655032, 1.780544, 
    1.191910, 1.048375, 1.107417, 1.084481, 1.046919, 1.076721, 1.087855, 1.093568, 1.188752, 1.275659, 1.582505, 1.701532, 
    1.144203, 1.030075, 1.090073, 1.051503, 1.008033, 1.018225, 1.023892, 1.045168, 1.091808, 1.141059, 1.535088, 1.641493, 
    1.090515, 1.008162, 1.056620, 1.019350, 1.008033, 1.018225, 1.023892, 1.045168, 1.038381, 1.089719, 1.490356, 1.611928, 
    1.049441, 1.008162, 1.023079, 1.019350, 1.008033, 1.018225, 1.023892, 1.045168, 1.003637, 1.052351, 1.417169, 1.565521, 
    1.018639, 1.008162, 1.008624, 1.019350, 1.008033, 1.018225, 1.023892, 1.045168, 1.003637, 1.028641, 1.362702, 1.519538, 
    1.018639, 1.008162, 1.008624, 1.019350, 1.008033, 1.018225, 1.023892, 1.045168, 1.003637, 1.011055, 1.313330, 1.490415, 
    1.018639, 1.008162, 1.008624, 1.019350, 1.008033, 1.018225, 1.023892, 1.045168, 1.003637, 1.011055, 1.261949, 1.468802, 
    1.018639, 1.008162, 1.008624, 1.019350, 1.008033, 1.018225, 1.023892, 1.045168, 1.003637, 1.011055, 1.224693, 1.449604, 
    1.018639, 1.008162, 1.008624, 1.019350, 1.008033, 1.018225, 1.023892, 1.045168, 1.003637, 1.011055, 1.186018, 1.424487, 
    1.018639, 1.008162, 1.008624, 1.019350, 1.008033, 1.018225, 1.023892, 1.045168, 1.003637, 1.011055, 1.134977, 1.365419, 
    1.018639, 1.008162, 1.008624, 1.019350, 1.008033, 1.018225, 1.023892, 1.045168, 1.003637, 1.011055, 1.096907, 1.285599
    ])
