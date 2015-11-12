## example cfg to dump HCAL conditions from the database
## (can be also used to dump sqlite content or to test fake conditions reading in CMSSW)
## Radek Ofierzynski, 9.11.2008
##
## Gena Kukartsev, 29.07.2009

import FWCore.ParameterSet.Config as cms

process = cms.Process("DUMP")

process.load("CondCore.CondDB.CondDB_cfi")

## specify which conditions you would like to dump to a text file in the "dump" vstring
process.prod = cms.EDAnalyzer("HcalDumpConditions",
                            dump = cms.untracked.vstring(
    'ChannelQuality' 
    ),
    outFilePrefix = cms.untracked.string('DumpCond')
)

process.hcal_db_producer = cms.ESProducer("HcalDbProducer",
    dump = cms.untracked.vstring(''),
    file = cms.untracked.string('')
)

## specify for which run you would like to get the conditions in the "firstRun"
process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(10000)
)


process.es_pool = cms.ESSource("PoolDBESSource",
     process.CondDB,
     timetype = cms.string('runnumber'),
     #connect = cms.string('sqlite_file:testExample.db'),
     #connect = cms.string('sqlite_file:test_o2o.db'),
     connect = cms.string('sqlite_file:testDropbox.db'),
     authenticationMethod = cms.untracked.uint32(0),
     toGet = cms.VPSet(
        cms.PSet(
            record = cms.string('HcalChannelQualityRcd'),
            #tag = cms.string('hcal_channelStatus_trivial_mc')
            tag = cms.string('gak_v2')
        ) 
)
)


# specify which conditions should be taken for input, 
# you can mix different es_sources as long as it's unique for each object
#process.es_pool = cms.ESSource(
#     "PoolDBESSource",
#     process.CondDB,
#     timetype = cms.string('runnumber'),
#     connect = cms.string('frontier://cmsfrontier.cern.ch:8000/FrontierProd/CMS_COND_31X_HCAL'),
#     authenticationMethod = cms.untracked.uint32(0),
#     toGet = cms.VPSet(
#         cms.PSet(
#             record = cms.string('HcalChannelQualityRcd'),
#             tag = cms.string('HcalChannelQuality_v1.02_mc')
#             )
#         )
# )

process.es_hardcode = cms.ESSource("HcalHardcodeCalibrations",
     toGet = cms.untracked.vstring('Pedestals', 
         'ZSThresholds')
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
#         )
#         )
# )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.p = cms.Path(process.prod)


