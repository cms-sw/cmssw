import FWCore.ParameterSet.Config as cms

process = cms.Process("DUMP")

process.load("CondCore.DBCommon.CondDBSetup_cfi")

## specify which conditions you would like to dump to a text file in the "dump" vstring
process.prod = cms.EDAnalyzer("CastorDumpConditions",
    dump = cms.untracked.vstring(
        #'Pedestals',
#        'PedestalWidths', 
#        'Gains', 
#        'QIEData', 
#        'ElectronicsMap',
#        'ChannelQuality', 
#        'GainWidths',
#	'RecoParams',
	'SaturationCorrs'
                                 ),
    outFilePrefix = cms.untracked.string('DumpCastorCond')
)


process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(1)
)

#process.CastorDbProducer = cms.ESProducer("CastorDbProducer")
#process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
#process.GlobalTag.globaltag = 'GR_R_61_V2::All'


process.es_pool = cms.ESSource(
   "PoolDBESSource",
   process.CondDBSetup,
   timetype = cms.string('runnumber'),
    connect = cms.string('sqlite_fle:CastorSaturationCorrs.db'),
authenticationMethod = cms.untracked.uint32(0),
   toGet = cms.VPSet(
       #cms.PSet(
#           record = cms.string('CastorPedestalsRcd'),
#           tag = cms.string('castor_pedestals_v1.0_test')
#           ),
#       cms.PSet(
#           record = cms.string('CastorPedestalWidthsRcd'),
#           tag = cms.string('castor_widths_v1.0_test')
#           ),
#       cms.PSet(
#           record = cms.string('CastorGainsRcd'),
#           tag = cms.string('castor_gains_v1.0_test')
#           ),
#       cms.PSet(
#           record = cms.string('CastorGainWidthsRcd'),
#           tag = cms.string('castor_gainwidths_v1.0_test')
#           ),
#       cms.PSet(
#           record = cms.string('CastorQIEDataRcd'),
#           tag = cms.string('castor_qie_v1.0_test')
#           ),
#       cms.PSet(
#           record = cms.string('CastorChannelQualityRcd'),
#           tag = cms.string('castor_channelstatus_v1.0_test')
#           ),
#       cms.PSet(
#           record = cms.string('CastorElectronicsMapRcd'),
#           tag = cms.string('castor_emap_v1.0_test')
#           ),
#       cms.PSet(
#           record = cms.string('CastorRecoParamsRcd'),
#           tag = cms.string('castor_recoparams_v1.00_test')
#           ),
       cms.PSet(
           record = cms.string('CastorSaturationCorrsRcd'),
           tag = cms.string('CastorSaturationCorrs_v1.00_offline')
           )
   )
)



#process.es_pool = cms.ESSource(
#   "PoolDBESSource",
#   process.CondDBSetup,
#   timetype = cms.string('runnumber'),
#   connect = cms.string('frontier://cmsfrontier.cern.ch:8000/FrontierProd/CMS_COND_31X_HCAL'),
#   authenticationMethod = cms.untracked.uint32(0),
#   toGet = cms.VPSet(
#       cms.PSet(
#           record = cms.string('CastorPedestalsRcd'),
#           tag = cms.string('castor_pedestals_v1.0_mc')
#           ),
#       cms.PSet(
#           record = cms.string('CastorPedestalWidthsRcd'),
#           tag = cms.string('castor_pedestalwidths_v1.0_mc')
#           ),
#       cms.PSet(
#           record = cms.string('CastorGainsRcd'),
#           tag = cms.string('castor_gains_v1.0_mc')
#           ),
#       cms.PSet(
#           record = cms.string('CastorGainWidthsRcd'),
#           tag = cms.string('castor_gainwidths_v1.0_mc')
#           ),
#       cms.PSet(
#           record = cms.string('CastorQIEDataRcd'),
#           tag = cms.string('castor_qie_v1.0')
#           ),
#       cms.PSet(
#           record = cms.string('CastorChannelQualityRcd'),
#           tag = cms.string('castor_channelquality_v1.0')
#           ),
#       cms.PSet(
#           record = cms.string('CastorElectronicsMapRcd'),
#           tag = cms.string('castor_emap_dcc_v1.0')
#           ),
#       cms.PSet(
#           record = cms.string('CastorRecoParamsRcd'),
#           tag = cms.string('castor_recoparams_v1.00_test')
#           ),
#       cms.PSet(
#           record = cms.string('CastorSaturationCorrsRcd'),
#           tag = cms.string('castor_saturationcorrs_v1.00_test')
#           )
#   )
#)


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.p = cms.Path(process.prod)



