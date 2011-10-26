## example cfg to dump HCAL conditions from the database
## (can be also used to dump sqlite content or to test fake conditions reading in CMSSW)
## Radek Ofierzynski, 9.11.2008
##
## Gena Kukartsev, July 29, 2009
## Gena Kukartsev, September 21, 2009

import FWCore.ParameterSet.Config as cms

process = cms.Process("DUMP")

process.load("CondCore.DBCommon.CondDBSetup_cfi")

## specify which conditions you would like to dump to a text file in the "dump" vstring
process.prod = cms.EDAnalyzer("HcalDumpConditions",
    dump = cms.untracked.vstring(
#        'Pedestals'
#        ,'PedestalWidths' 
#        ,'Gains' 
#        ,'QIEData' 
#        ,'ElectronicsMap'
#        ,'ChannelQuality' 
#        ,'GainWidths' 
#        ,'RespCorrs' 
#        ,'TimeCorrs'
#        ,'LUTCorrs'
#        ,'PFCorrs'
#        'L1TriggerObjects'
#        ,'ZSThresholds'
#        ,'ValidationCorrs' 
#        ,'LutMetadata'
#        ,'DcsValues'
#        ,'DcsMap'
        'TimingParams'
#    'RecoParams'
#    ,'LongRecoParams'
#    ,'MCParams'
#    ,'FlagHFDigiTimeParams'
        ),
    outFilePrefix = cms.untracked.string('DumpCond')
)

## specify for which run you would like to get the conditions in the "firstRun"
process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(1)
)


process.es_pool = cms.ESSource("PoolDBESSource",
     process.CondDBSetup,
     timetype = cms.string('runnumber'),
     connect = cms.string('sqlite_file:testExample.db'),
     authenticationMethod = cms.untracked.uint32(0),
     toGet = cms.VPSet(
#        cms.PSet(
#            record = cms.string('HcalPedestalsRcd'),
#            tag = cms.string('hcal_pedestals_fC_v6_mc')
#        ), 
#        cms.PSet(
#            record = cms.string('HcalPedestalWidthsRcd'),
#            tag = cms.string('hcal_widths_fC_v6_mc')
#        ), 
#        cms.PSet(
#            record = cms.string('HcalGainsRcd'),
#            tag = cms.string('hcal_gains_v3.01_physics_mc')
#        ), 
#        cms.PSet(
#            record = cms.string('HcalQIEDataRcd'),
#            tag = cms.string('qie_normalmode_v6.01')
#        ), 
#        cms.PSet(
#            record = cms.string('HcalChannelQualityRcd'),
#            tag = cms.string('hcal_channelStatus_trivial_mc')
#        ), 
#        cms.PSet(
#            record = cms.string('HcalRespCorrsRcd'),
#            tag = cms.string('hcal_respcorr_trivial_v1.01_mc')
#        ), 
#        cms.PSet(
#            record = cms.string('HcalL1TriggerObjectsRcd'),
#            tag = cms.string('hcal_L1TriggerObject_trivial_mc')
#        ), 
#        cms.PSet(
#            record = cms.string('HcalElectronicsMapRcd'),
#            tag = cms.string('official_emap_v7.00')
#        ),
#        cms.PSet(
#            record = cms.string('HcalValidationCorrsRcd'),
#            tag = cms.string('hcal_validationcorr_trivial_v1.01_mc')
#        ),
#        cms.PSet(
#            record = cms.string('HcalLutMetadataRcd'),
#            tag = cms.string('hcal_lutmetadata_trivial_v1.01_mc')
#        ) 
#        cms.PSet(
#            record = cms.string('HcalDcsMapRcd'),
#            tag = cms.string('HcalDcsMap_v1.00_test')
#        )
        cms.PSet(
            record = cms.string('HcalTimingParamsRcd'),
            tag = cms.string('hcal_timingparams_v1.00_test')
        )    
#        cms.PSet(
#            record = cms.string('HcalRecoParamsRcd'),
#            tag = cms.string('hcal_recoparams_v1.00_test')
#        ), 
#        cms.PSet(
#            record = cms.string('HcalLongRecoParamsRcd'),
#            tag = cms.string('hcal_longrecoparams_v1.00_test')
#        ), 
#        cms.PSet(
#            record = cms.string('HcalMCParamsRcd'),
#            tag = cms.string('hcal_mcparams_v1.00_test')
#        ),
#        cms.PSet(
#    record = cms.string('HcalFlagHFDigiTimeParamsRcd'),
#    tag = cms.string('hcal_flaghfdigitimeparams_v1.00_test')
#    ),
        
        
)
)


## specify which conditions should be taken for input, 
## you can mix different es_sources as long as it's unique for each object
# process.es_pool = cms.ESSource(
#     "PoolDBESSource",
#     process.CondDBSetup,
#     timetype = cms.string('runnumber'),
#     connect = cms.string('frontier://FrontierDev/CMS_COND_HCAL'),
#     authenticationMethod = cms.untracked.uint32(0),
#     toGet = cms.VPSet(
#         cms.PSet(
#             record = cms.string('HcalPedestalsRcd'),
#             tag = cms.string('hcal_pedestals_fC_v3_mc')
#             ), 
#         cms.PSet(
#             record = cms.string('HcalPedestalWidthsRcd'),
#             tag = cms.string('hcal_widths_fC_v3_mc')
#             ), 
#         cms.PSet(
#             record = cms.string('HcalGainsRcd'),
#             tag = cms.string('hcal_gains_v2_physics_50_mc')
#             ), 
#         cms.PSet(
#             record = cms.string('HcalQIEDataRcd'),
#             tag = cms.string('qie_normalmode_v5_mc')
#             ), 
#         cms.PSet(
#             record = cms.string('HcalElectronicsMapRcd'),
#             tag = cms.string('official_emap_v5_080208_mc')
#             ),
#         cms.PSet(
#             record = cms.string('HcalLutMetadataRcd'),
#             tag = cms.string('hcal_lutmetadata_trivial_v1.01_mc')
#         ) 
#         cms.PSet(
#             record = cms.string('HcalDcsMapRcd'),
#             tag = cms.string('HcalDcsMap_v1.00_test')
#         ) 
#         )
# )

process.es_hardcode = cms.ESSource("HcalHardcodeCalibrations",
     toGet = cms.untracked.vstring(
#        'Pedestals'
#        ,'PedestalWidths' 
#        ,'Gains' 
#        ,'QIEData' 
#        ,'ElectronicsMap'
#        ,'ChannelQuality' 
#        ,'GainWidths' 
#        ,'RespCorrs' 
#        ,'TimeCorrs'
#        ,'LUTCorrs'
#        ,'PFCorrs'
#        ,'L1TriggerObjects'
#        ,'ZSThresholds'
#        ,'ValidationCorrs' 
#        ,'LutMetadata'
#        ,'DcsValues'
#        ,'DcsMap'
        'TimingParams'
#        ,'RecoParams'
#        ,'LongRecoParams'
#        ,'MCParams'
#        ,'FlagHFDigiTimeParams'        
        )
 )

#process.es_ascii = cms.ESSource("HcalTextCalibrations",
#     input = cms.VPSet(
#         cms.PSet(
#             object = cms.string('Pedestals'),
#             file = cms.FileInPath('CondFormats/HcalObjects/data/hcal_pedestals_fC_v5.txt')
#             ), 
#         cms.PSet(
#             object = cms.string('PedestalWidths'),
#             file = cms.FileInPath('CondFormats/HcalObjects/data/hcal_widths_fC_v5.txt')
#         ), 
#         cms.PSet(
#             object = cms.string('Gains'),
#             file = cms.FileInPath('CondFormats/HcalObjects/data/hcal_gains_v1.txt')
#         ), 
#         cms.PSet(
#             object = cms.string('GainWidths'),
#             file = cms.FileInPath('CondFormats/HcalObjects/data/hcal_gains_widths_v1.txt')
#         ), 
#         cms.PSet(
#             object = cms.string('QIEData'),
#             file = cms.FileInPath('CondFormats/HcalObjects/data/qie_normalmode_v6_cand2_fakeZDC.txt')
#         ), 
#         cms.PSet(
#             object = cms.string('ElectronicsMap'),
#             file = cms.FileInPath('CondFormats/HcalObjects/data/official_emap_v7.00_081109.txt')
#         ), 
#         cms.PSet(
#             object = cms.string('ChannelQuality'),
#             file = cms.FileInPath('CondFormats/HcalObjects/data/hcal_channelStatus_default.txt')
#         ),
#         cms.PSet(
#             object = cms.string('RespCorrs'),
#             file = cms.FileInPath('CondFormats/HcalObjects/data/hcal_respCorr_trivial_HF0.7.txt')
#         ) ,
#         cms.PSet(
#             object = cms.string('L1TriggerObjects'),
#             file = cms.FileInPath('CondFormats/HcalObjects/data/hcal_L1TriggerObject_trivial.txt')
#         ) ,
#         cms.PSet(
#             object = cms.string('ValidationCorrs'),
#             file = cms.FileInPath('CondFormats/HcalObjects/data/hcal_validationCorr_trivial_HF0.7.txt')
#         ),
#         cms.PSet(
#             object = cms.string('LutMetadata'),
#             tag = cms.FileInPath('CondFormats/HcalObjects/data/hcal_lutmetadata_trivial_v1.01_mc')
#         ), 
#         cms.PSet(
#             object = cms.string('DcsMap'),
#             file = cms.FileInPath('HcalDcsMap_v1.00_test')
#         ) 
#         cms.PSet(
#             object = cms.string('RecoParams'),
#             file = cms.FileInPath('CondTools/Hcal/test/testdata/RecoParams2011-run153943.txt')
#         ), 
#         cms.PSet(
#             object = cms.string('LongRecoParams'),
#             file = cms.FileInPath('CondTools/Hcal/test/testdata/LongRecoParams2011-run153943.txt')
#         ), 
#         cms.PSet(
#             object = cms.string('MCParams'),
#             file = cms.FileInPath('CondTools/Hcal/test/testdata/MCParams.txt')
#         ) ,
#         cms.PSet(
#             object = cms.string('FlagHFDigiTimeParams'),
#             file = cms.FileInPath('CondTools/Hcal/test/testdata/HcalFlagHFDigiTimeParams.txt')
#         ) ,
#         )
# )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.p = cms.Path(process.prod)


