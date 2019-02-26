import FWCore.ParameterSet.Config as cms

process = cms.Process("ProdTPGParam")

# Calo geometry service model
process.load("Geometry.CaloEventSetup.CaloGeometry_cfi")
process.load("Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi")
process.load("Geometry.HcalCommonData.hcalDDConstants_cff")
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

# ecal mapping
process.load("Geometry.EcalMapping.EcalMapping_cfi")
process.eegeom = cms.ESSource("EmptyESSource",
        recordName = cms.string('EcalMappingRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

process.load("CondCore.CondDB.CondDB_cfi")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '101X_postLS2_realistic_v6', '')

process.GlobalTag.toGet = cms.VPSet(
  cms.PSet(record = cms.string('EcalPedestalsRcd'),
           tag = cms.string('EcalPedestals_mid2021_235fb_mc'),
           ),
  cms.PSet(record = cms.string('EcalIntercalibConstantsRcd'),
           tag = cms.string('EcalIntercalibConstants_TL235fb_mc'),
           ),
  cms.PSet(record = cms.string('EcalLaserAPDPNRatiosRcd'),
           tag = cms.string('EcalLaserAPDPNRatios_TL235fb_mc'),
           ),
  cms.PSet(record = cms.string('EcalLaserAlphasRcd'),
           tag = cms.string('EcalLaserAlphas_EB_sic1_btcp1_EE_sic1_btcp1'),
           )
)
#########################
process.source = cms.Source("EmptySource",
       firstRun = cms.untracked.uint32(161310)
       ##firstRun = cms.untracked.uint32(257385)                                                             
)


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.TPGParamProducer = cms.EDAnalyzer("EcalTPGParamBuilder",

    #### inputs/ouputs control ####
    ##writeToDB  = cms.bool(True),
    ##writeTODB = cms.bool(False),                                                                       
    writeToDB  = cms.bool(False),
    allowDBEE  = cms.bool(False),

    DBsid   = cms.string('cms_omds_adg'), ## real DB 
    ##DBsid   = cms.string('int2r'), ## test DB
    ## P5 online DB
    DBuser  = cms.string('CMS_ECAL_R'),
    DBpass  = cms.string('3c4l_r34d3r'),
    ##DBpass  = cms.string('*******'),
    ## test DB
    ##DBuser  = cms.string('cms_ecal_conf_test'),
    DBport  = cms.uint32(10121),

    TPGWritePed = cms.uint32(1), # can be 1=load ped from offline DB  0=use previous ped NN=use ped from ped_conf_id=NN
    TPGWriteLin = cms.uint32(1),
    TPGWriteSli = cms.uint32(1),
    TPGWriteWei = cms.uint32(1),
    TPGWriteLut = cms.uint32(1),
    TPGWriteFgr = cms.uint32(1),
    TPGWriteSpi = cms.uint32(1),
    TPGWriteDel = cms.uint32(1),
    TPGWriteBxt = cms.uint32(0), # these can be 0=use same as existing number for this tag or NN=use badxt from bxt_conf_id=NN
    TPGWriteBtt = cms.uint32(0), 
    TPGWriteBst = cms.uint32(0), 

    writeToFiles = cms.bool(True),
                      
    outFile = cms.string('TPG_235fb.txt'),
                            
                                          
   #### TPG config tag and version (if not given it will be automatically given ) ####
    TPGtag = cms.string('BEAMV6_TRANS_SPIKEKILL'),
    TPGversion = cms.uint32(1),
                                        
   #### TPG calculation parameters ####
    useTransverseEnergy = cms.bool(True),    ## true when TPG computes transverse energy, false for energy
    Et_sat_EB = cms.double(128.0),            ## Saturation value (in GeV) of the TPG before the compressed-LUT (rem: with 35.84 the TPG_LSB = crystal_LSB)
    Et_sat_EE = cms.double(128.0),            ## Saturation value (in GeV) of the TPG before the compressed-LUT (rem: with 35.84 the TPG_LSB = crystal_LSB)

    sliding = cms.uint32(0),                 ## Parameter used for the FE data format, should'nt be changed

    weight_timeShift = cms.double(0.),       ## weights are computed shifting the timing of the shape by this amount in ns: val>0 => shape shifted to the right
    weight_sampleMax = cms.uint32(3),        ## position of the maximum among the 5 samples used by the TPG amplitude filter
    weight_unbias_recovery = cms.bool(True), ## true if weights after int conversion are forced to have sum=0. Pb, in that case it can't have sum f*w = 1

    forcedPedestalValue = cms.int32(-3),     ## use this value instead of getting it from DB or MC
                                             ## -1: means use value from DB or MC.
                                             ## -2: ped12 = 0 used to cope with FENIX bug
                                             ## -3: used with sFGVB: baseline subtracted is pedestal-offset*sin(theta)/G with G=mult*2^-(shift+2) 
    pedestal_offset =  cms.uint32(150),      ## pedestal offset used with option forcedPedestalValue = -3

    useInterCalibration = cms.bool(True),    ## use or not values from DB. If not, 1 is assumed

    timing_delays_EB = cms.string('Delays_EB.txt'), # timing delays for latency EB / TT 
    timing_delays_EE = cms.string('Delays_EE.txt'), # timing delays for latency EE / strip                                         
    timing_phases_EB = cms.string('Phases_EB.txt'), # TCC phase setting for EB / TT
    timing_phases_EE = cms.string('Phases_EE.txt'), # TCC phase setting for EE / strip

    useTransparencyCorr = cms.bool(True),   ## true if you want to correct TPG for transparency change in EE                                          
#    transparency_corrections = cms.string('hourly_235'), # transparency corr to be used to compute linearizer parameters 1/crystal
    transparency_corrections = cms.string('tag'), # transparency corr to be used to compute linearizer parameters 1/crystal
                                          
                                          
    SFGVB_Threshold = cms.uint32(16),             ## (adc) SFGVB threshold in FE
    SFGVB_lut = cms.uint32(0xfffefee8),           ## SFGVB LUT in FE                                
    SFGVB_SpikeKillingThreshold = cms.int32(16),  ## (GeV) Spike killing threshold applied in TPG ET in TCC (-1 no killing)
                                        
    forceEtaSlice = cms.bool(False),         ## when true, same linearization coeff for all crystals belonging to a given eta slice (tower)

    LUT_option = cms.string('Linear'),       ## compressed LUT option can be: "Identity", "Linear", "EcalResolution"
    LUT_threshold_EB = cms.double(0.250),    ## All Trigger Primitives <= threshold (in GeV) will be set to 0 
    LUT_threshold_EE = cms.double(0.250),    ## All Trigger Primitives <= threshold (in GeV) will be set to 0 
    LUT_stochastic_EB = cms.double(0.03),    ## Stochastic term of the ECAL-EB ET resolution (used only if LUT_option="EcalResolution")
    LUT_noise_EB = cms.double(0.2),          ## noise term (GeV) of the ECAL-EB ET resolution (used only if LUT_option="EcalResolution")
    LUT_constant_EB = cms.double(0.005),     ## constant term of the ECAL-EB ET resolution (used only if LUT_option="EcalResolution")
    LUT_stochastic_EE = cms.double(0.03),    ## Stochastic term of the ECAL-EE ET resolution (used only if LUT_option="EcalResolution")
    LUT_noise_EE = cms.double(0.2),          ## noise term (GeV) of the ECAL-EE ET resolution (used only if LUT_option="EcalResolution")
    LUT_constant_EE = cms.double(0.005),     ## constant term of the ECAL-EE ET resolution (used only if LUT_option="EcalResolution")

    TTF_lowThreshold_EB = cms.double(2.0),   ## EB Trigger Tower Flag low threshold in GeV                                                                                                                   
    TTF_highThreshold_EB = cms.double(4.0),  ## EB Trigger Tower Flag high threshold in GeV                                                                                                                  
    TTF_lowThreshold_EE = cms.double(2.0),   ## EE Trigger Tower Flag low threshold in GeV                                                                                                                   
    TTF_highThreshold_EE = cms.double(4.0),  ## EE Trigger Tower Flag high threshold in GeV                                                                                                                   
    FG_lowThreshold_EB = cms.double(3.9),      ## EB Fine Grain Et low threshold in GeV
    FG_highThreshold_EB = cms.double(3.9),     ## EB Fine Grain Et high threshold in GeV
    FG_lowRatio_EB = cms.double(0.9),          ## EB Fine Grain low-ratio
    FG_highRatio_EB = cms.double(0.9),         ## EB Fine Grain high-ratio

    FG_lut_EB = cms.uint32(0x08),              ## EB Fine Grain Look-up table. Put something != 0 if you really know what you do!
    FG_Threshold_EE = cms.double(18.75),       ## EE Fine threshold in GeV
    FG_lut_strip_EE = cms.uint32(0xfffefee8),  ## EE Fine Grain strip Look-up table
    FG_lut_tower_EE = cms.uint32(0)            ## EE Fine Grain tower Look-up table
)

process.p = cms.Path(process.TPGParamProducer)

