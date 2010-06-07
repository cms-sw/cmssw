import FWCore.ParameterSet.Config as cms

process = cms.Process("ProdTPGParam")

# Calo geometry service model
process.load("Geometry.CaloEventSetup.CaloGeometry_cfi")
process.load("Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi")
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

# ecal mapping
process.load("Geometry.EcalMapping.EcalMapping_cfi")
process.eegeom = cms.ESSource("EmptyESSource",
    recordName = cms.string('EcalMappingRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

# Get hardcoded conditions the same used for standard digitization before CMSSW_3_1_x
process.load("CalibCalorimetry.EcalTrivialCondModules.EcalTrivialCondRetriever_cfi")


#########################
process.source = cms.Source("EmptySource",
       firstRun = cms.untracked.uint32(100000000) ### need to use latest run to pick-up update values from DB 
)


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.TPGParamProducer = cms.EDFilter("EcalTPGParamBuilder",

    #### inputs/ouputs control ####
    writeToDB  = cms.bool(False),
    allowDBEE  = cms.bool(True),

    DBsid   = cms.string('cms_omds_lb'),
    DBuser  = cms.string('cms_ecal_conf'),
    DBpass  = cms.string('0r4cms_3c4lc0nf'),
    DBport  = cms.uint32(10121),

    TPGWritePed = cms.uint32(1),
    TPGWriteLin = cms.uint32(1),
    TPGWriteSli = cms.uint32(1),
    TPGWriteWei = cms.uint32(1),
    TPGWriteLut = cms.uint32(1),
    TPGWriteFgr = cms.uint32(1),
    TPGWriteBxt = cms.uint32(0),
    TPGWriteBtt = cms.uint32(0), #do not change

    writeToFiles = cms.bool(True),
    outFile = cms.string('TPG_H2.txt'),
   #### TPG config tag and version (if not given it will be automatically given ) ####
    TPGtag = cms.string('H2'),
    TPGversion = cms.uint32(1),
                                        
   #### TPG calculation parameters ####
    useTransverseEnergy = cms.bool(False),   ## true when TPG computes transverse energy, false for energy
    Et_sat_EB = cms.double(256.0),           ## Saturation value (in GeV) of the TPG before the compressed-LUT (rem: with 35.84 the TPG_LSB = crystal_LSB)
    Et_sat_EE = cms.double(256.0),           ## Saturation value (in GeV) of the TPG before the compressed-LUT (rem: with 35.84 the TPG_LSB = crystal_LSB)

    sliding = cms.uint32(0),                 ## Parameter used for the FE data format, should'nt be changed

    weight_timeShift = cms.double(0.),       ## weights are computed shifting the timing of the shape by this amount in ns: val>0 => shape shifted to the right
    weight_sampleMax = cms.uint32(3),        ## position of the maximum among the 5 samples used by the TPG amplitude filter
    weight_unbias_recovery = cms.bool(True), ## true if weights after int conversion are forced to have sum=0. Pb, in that case it can't have sum f*w = 1

    forcedPedestalValue = cms.int32(-2),     ## use this value instead of getting it from DB or MC (-1 means use DB or MC. -2 used to cope with FENIX bug)
                                             ## -1: means use value from DB or MC.
                                             ## -2: ped12 = 0 used to cope with FENIX bug
                                             ## -3: used with sFGVB: baseline subtracted is pedestal-offset*sin(theta)/G with G=mult*2^-(shift+2) 
    pedestal_offset =  cms.uint32(300),      ## pedestal offset used with option forcedPedestalValue = -3

    useInterCalibration = cms.bool(False),   ## use or not values from DB. If not, 1 is assumed

    SFGVB_Threshold = cms.uint32(0),         ## used with option forcedPedestalValue = -3
    SFGVB_lut = cms.uint32(0),               ## used with option forcedPedestalValue = -3                                

    forceEtaSlice = cms.bool(True),        ## when true, same linearization coeff for all crystals belonging to a given eta slice (tower)

    LUT_option = cms.string('Linear'),      ## compressed LUT option can be: "Identity", "Linear", "EcalResolution"
    LUT_threshold_EB = cms.double(0.250),   ## All Trigger Primitives <= threshold (in GeV) will be set to 0 
    LUT_threshold_EE = cms.double(0.250),   ## All Trigger Primitives <= threshold (in GeV) will be set to 0 
    LUT_stochastic_EB = cms.double(0.03),   ## Stochastic term of the ECAL-EB ET resolution (used only if LUT_option="EcalResolution")
    LUT_noise_EB = cms.double(0.2),         ## noise term (GeV) of the ECAL-EB ET resolution (used only if LUT_option="EcalResolution")
    LUT_constant_EB = cms.double(0.005),    ## constant term of the ECAL-EB ET resolution (used only if LUT_option="EcalResolution")
    LUT_stochastic_EE = cms.double(0.03),   ## Stochastic term of the ECAL-EE ET resolution (used only if LUT_option="EcalResolution")
    LUT_noise_EE = cms.double(0.2),         ## noise term (GeV) of the ECAL-EE ET resolution (used only if LUT_option="EcalResolution")
    LUT_constant_EE = cms.double(0.005),    ## constant term of the ECAL-EE ET resolution (used only if LUT_option="EcalResolution")

    TTF_lowThreshold_EB = cms.double(1.0),   ## EB Trigger Tower Flag low threshold in GeV
    TTF_highThreshold_EB = cms.double(2.0),  ## EB Trigger Tower Flag high threshold in GeV
    TTF_lowThreshold_EE = cms.double(1.0),   ## EE Trigger Tower Flag low threshold in GeV
    TTF_highThreshold_EE = cms.double(2.0),  ## EE Trigger Tower Flag high threshold in GeV

    FG_lowThreshold_EB = cms.double(0.3125),   ## EB Fine Grain Et low threshold in GeV
    FG_highThreshold_EB = cms.double(0.3125),  ## EB Fine Grain Et high threshold in GeV
    FG_lowRatio_EB = cms.double(0.8),          ## EB Fine Grain low-ratio
    FG_highRatio_EB = cms.double(0.8),         ## EB Fine Grain high-ratio
    FG_lut_EB = cms.uint32(0x08),              ## EB Fine Grain Look-up table. Put something != 0 if you really know what you do!
    FG_Threshold_EE = cms.double(0.0),         ## EE Fine threshold in GeV
    FG_lut_strip_EE = cms.uint32(0),           ## EE Fine Grain strip Look-up table
    FG_lut_tower_EE = cms.uint32(0)            ## EE Fine Grain tower Look-up table
)

process.p = cms.Path(process.TPGParamProducer)

