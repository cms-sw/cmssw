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

#================================================================================================
#================================================================================================
# Get hardcoded conditions the same used for standard digitization
process.load("CalibCalorimetry.EcalTrivialCondModules.EcalTrivialCondRetriever_cfi")
process.es_prefer = cms.ESPrefer("PoolDBESSource","ecalConditions")

from CondCore.DBCommon.CondDBSetup_cfi import *
process.ecalConditions = cms.ESSource("PoolDBESSource",
    CondDBSetup,
    siteLocalConfig = cms.untracked.bool(True),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('EcalPedestalsRcd'),
        tag = cms.string('EcalPedestals_mc')
    ), 

        cms.PSet(
            record = cms.string('EcalADCToGeVConstantRcd'),
            tag = cms.string('EcalADCToGeVConstant_EBg50_EEwithB_new')
        ), 
        cms.PSet(
            record = cms.string('EcalChannelStatusRcd'),
            tag = cms.string('EcalChannelStatus_mc')
        ), 
        cms.PSet(
            record = cms.string('EcalGainRatiosRcd'),
            tag = cms.string('EcalGainRatios_mc')
        ), 
        cms.PSet(
            record = cms.string('EcalIntercalibConstantsRcd'),
            tag = cms.string('EcalIntercalibConstants_EBg50_EEwithB_new')
        ), 
        ###cms.PSet(
        ###    record = cms.string('EcalWeightXtalGroupsRcd'),
        ###    tag = cms.string('EcalWeightXtalGroups_mc')
        ###), 
        ###cms.PSet(
        ###    record = cms.string('EcalTBWeightsRcd'),
        ###    tag = cms.string('EcalTBWeights_mc')
        ###), 
        ###cms.PSet(
        ###    record = cms.string('EcalLaserAlphasRcd'),
        ###    tag = cms.string('EcalLaserAlphas_mc')
        ###), 
        ###cms.PSet(
        ###    record = cms.string('EcalLaserAPDPNRatiosRcd'),
        ###    ###record = cms.string('EcalLaserAPDPNRatiosRcd'),
        ###    ###tag = cms.string('EcalLaserAPDPNRatios_mc')
        ###    ###tag = cms.string('EcalLaserAPDPNRatios_online')
        ###    tag = cms.string('EcalLaserAPDPNRatios_v2_online')
        ###), 
        ###cms.PSet(
        ###    record = cms.string('EcalLaserAPDPNRatiosRefRcd'),
        ###    tag = cms.string('EcalLaserAPDPNRatiosRef_mc')
        ###)
        ),
    messagelevel = cms.untracked.uint32(0),
    timetype = cms.untracked.string('timestamp'),
    ##timetype = cms.string('runnumber'),
    #connect = cms.string('frontier://cms_conditions_data/CMS_COND_ECAL'), ##cms_conditions_data/CMS_COND_ECAL"
    ##connect = cms.string('frontier://FrontierPrep/CMS_COND_ECAL_LT'), ##cms_conditions_data/CMS_COND_ECAL"
    #####connect = cms.string('frontier://cms_conditions_data/CMS_COND_31X_ECAL'),
    #####connect = cms.string('oracle://cms_orcon_prod/CMS_COND_31X_ECAL'),
    connect = cms.string('frontier://FrontierProd/CMS_COND_31X_ECAL'),
    ##connect = cms.string('oracle://cms_orcoff_prep/CMS_COND_ECAL_LT'),

    authenticationMethod = cms.untracked.uint32(1)
)
#================================================================================================
#================================================================================================

### Get hardcoded conditions the same used for standard digitization before CMSSW_3_1_x
####process.load("CalibCalorimetry.EcalTrivialCondModules.EcalTrivialCondRetriever_cfi")
### or Get DB parameters 
##process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
##process.GlobalTag.globaltag = "IDEAL_31X::All"

##from CondCore.DBCommon.CondDBSetup_cfi import *
##CondDBSetup.DBParameters.authenticationPath = '/afs/cern.ch/cms/DB/conddb'
##process.ecalIntercalibConstants = cms.ESSource("PoolDBESSource",CondDBSetup,
##        connect = cms.string("oracle://cms_orcoff_prep/CMS_COND_31X_ALL"),
##        toGet = cms.VPSet( cms.PSet(
##                record = cms.string("EcalIntercalibConstantsRcd"),
##                tag = cms.string("EcalIntercalibConstants_avg1_ideal")
##                ) )
##)
##process.es_prefer_ecalIntercalibConstants = cms.ESPrefer("PoolDBESSource","ecalIntercalibConstants")

#####CondDBSetup.DBParameters.authenticationPath = '/afs/cern.ch/cms/DB/conddb'
#####process.ecalADCToGeV = cms.ESSource("PoolDBESSource",CondDBSetup,
#####        connect = cms.string("oracle://cms_orcoff_prep/CMS_COND_31X_ALL"),
#####        toGet = cms.VPSet( cms.PSet(
#####                record = cms.string("EcalADCToGeVConstantRcd"),
#####                tag = cms.string("EcalADCToGeVConstant_avg1_ideal")
#####                ) )
#####)
#####process.es_prefer_ecalADCToGeV = cms.ESPrefer("PoolDBESSource","ecalADCToGeV")

####process.load("CondCore.DBCommon.CondDBCommon_cfi")
#####process.CondDBCommon.connect = 'sqlite_file:DB.db'
####process.CondDBCommon.connect = 'oracle://cms_orcoff_prep/CMS_COND_31X_ALL'
####process.CondDBCommon.DBParameters.authenticationPath = '/afs/cern.ch/cms/DB/conddb'


#########################
process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.TPGParamProducer = cms.EDFilter("EcalTPGParamBuilder",

    #### inputs/ouputs control ####
    writeToDB  = cms.bool(False),
    allowDBEE  = cms.bool(False),

    DBsid   = cms.string('ecalh4db'),
    DBuser  = cms.string('test09'),
    DBpass  = cms.string('oratest09'),
    DBport  = cms.uint32(1521),

    writeToFiles = cms.bool(True),
    outFile = cms.string('TPG_startup.txt'),
   #### TPG config tag and version (if not given it will be automatically given ) ####
    TPGtag = cms.string('CRAFT'),
    TPGversion = cms.uint32(1),
                                        
   #### TPG calculation parameters ####
    useTransverseEnergy = cms.bool(True),   ## true when TPG computes transverse energy, false for energy
    Et_sat_EB = cms.double(64.0),           ## Saturation value (in GeV) of the TPG before the compressed-LUT (rem: with 35.84 the TPG_LSB = crystal_LSB)
    Et_sat_EE = cms.double(64.0),           ## Saturation value (in GeV) of the TPG before the compressed-LUT (rem: with 35.84 the TPG_LSB = crystal_LSB)

    sliding = cms.uint32(0),                ## Parameter used for the FE data format, should'nt be changed

    weight_sampleMax = cms.uint32(3),       ## position of the maximum among the 5 samples used by the TPG amplitude filter

    forcedPedestalValue = cms.int32(-1),   ## use this value instead of getting it from DB or MC (-1 means use DB or MC)
    forceEtaSlice = cms.bool(False),         ## when true, same linearization coeff for all crystals belonging to a given eta slice (tower). Implemented only for EB

    LUT_option = cms.string('Linear'),      ## compressed LUT option can be: "Identity", "Linear", "EcalResolution"
    LUT_threshold_EB = cms.double(0.750),   ## All Trigger Primitives <= threshold (in GeV) will be set to 0 
    LUT_threshold_EE = cms.double(0.750),   ## All Trigger Primitives <= threshold (in GeV) will be set to 0 
    LUT_stochastic_EB = cms.double(0.03),   ## Stochastic term of the ECAL-EB ET resolution (used only if LUT_option="EcalResolution")
    LUT_noise_EB = cms.double(0.2),         ## noise term (GeV) of the ECAL-EB ET resolution (used only if LUT_option="EcalResolution")
    LUT_constant_EB = cms.double(0.005),    ## constant term of the ECAL-EB ET resolution (used only if LUT_option="EcalResolution")
    LUT_stochastic_EE = cms.double(0.03),   ## Stochastic term of the ECAL-EE ET resolution (used only if LUT_option="EcalResolution")
    LUT_noise_EE = cms.double(0.2),         ## noise term (GeV) of the ECAL-EE ET resolution (used only if LUT_option="EcalResolution")
    LUT_constant_EE = cms.double(0.005),    ## constant term of the ECAL-EE ET resolution (used only if LUT_option="EcalResolution")

    TTF_lowThreshold_EB = cms.double(1.0),  ## EB Trigger Tower Flag low threshold in GeV
    TTF_highThreshold_EB = cms.double(1.0), ## EB Trigger Tower Flag high threshold in GeV
    TTF_lowThreshold_EE = cms.double(1.0),  ## EE Trigger Tower Flag low threshold in GeV
    TTF_highThreshold_EE = cms.double(1.0), ## EE Trigger Tower Flag high threshold in GeV

    FG_lowThreshold_EB = cms.double(5.0),   ## EB Fine Grain Et low threshold in GeV
    FG_highThreshold_EB = cms.double(5.0),  ## EB Fine Grain Et high threshold in GeV
    FG_lowRatio_EB = cms.double(0.8),       ## EB Fine Grain low-ratio
    FG_highRatio_EB = cms.double(0.8),      ## EB Fine Grain high-ratio
    FG_lut_EB = cms.uint32(0),              ## EB Fine Grain Look-up table. Put something != 0 if you really know what you do!
    FG_Threshold_EE = cms.double(0.0),      ## EE Fine threshold in GeV
    FG_lut_strip_EE = cms.uint32(0),        ## EE Fine Grain strip Look-up table
    FG_lut_tower_EE = cms.uint32(0)         ## EE Fine Grain tower Look-up table
)

process.p = cms.Path(process.TPGParamProducer)

