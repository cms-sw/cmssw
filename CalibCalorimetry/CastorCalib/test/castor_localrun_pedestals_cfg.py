import FWCore.ParameterSet.Config as cms

process = cms.Process("PEDS")
process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.source = cms.Source("HcalTBSource",
    streams = cms.untracked.vstring('HCAL_Trigger', 
        'HCAL_DCC690','HCAL_DCC691','HCAL_DCC692', 
),
    fileNames = cms.untracked.vstring('/store/caft2/user/campbell/castor_localruns/USC_119814.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10000)
)

process.castorDigis = cms.EDProducer("CastorRawToDigi",
   CastorFirstFED = cms.untracked.int32(690),
   FilterDataQuality = cms.bool(True),
   ExceptionEmptyData = cms.untracked.bool(True),
   InputLabel = cms.InputTag("source"),
   UnpackCalib = cms.untracked.bool(False),
   FEDs = cms.untracked.vint32(690,691,692),
   lastSample = cms.int32(9),
   firstSample = cms.int32(0)
) 

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('USC_XXXXXX_unpacked.root')
)
process.dumpRaw = cms.EDAnalyzer( "DumpFEDRawDataProduct",

   feds = cms.untracked.vint32( 690,691,692,693 ),
   dumpPayload = cms.untracked.bool( True )
)

process.m = cms.EDAnalyzer("HcalDigiDump")

process.dump = cms.EDAnalyzer('HcalTBObjectDump',
                              hcalTBTriggerDataTag = cms.InputTag('tbunpack'),
                              hcalTBRunDataTag = cms.InputTag('tbunpack'),
                              hcalTBEventPositionTag = cms.InputTag('tbunpack'),
                              hcalTBTimingTag = cms.InputTag('tbunpack')
)

process.dumpECA = cms.EDAnalyzer("EventContentAnalyzer")

process.CastorDbProducer = cms.ESProducer("CastorDbProducer")

#process.es_hardcode = cms.ESSource("CastorHardcodeCalibrations",
    #toGet = cms.untracked.vstring('Gains', 
        #'Pedestals', 
        #'PedestalWidths', 
        #'GainWidths', 
        #'QIEShape', 
        #'QIEData', 
        #'ChannelQuality', 
        #'RespCorrs', 
        #'ZSThresholds')
#)

#process.es_ascii = cms.ESSource("CastorTextCalibrations",
   #input = cms.VPSet(cms.PSet(
       #object = cms.string('ElectronicsMap'),
       #file = cms.FileInPath('cmssw_emap_3dcc_v1.txt')
   #))
#)

# use this when it works
#   connect = cms.string('frontier://cmsfrontier.cern.ch:8000/FrontierProd/CMS_COND_31X_HCAL'),

process.es_pool = cms.ESSource(
   "PoolDBESSource",
   process.CondDBSetup,
   timetype = cms.string('runnumber'),
   connect = cms.string('frontier://cmsfrontier.cern.ch:8000/FrontierPrep/CMS_COND_30X_HCAL'),
   authenticationMethod = cms.untracked.uint32(0),
   toGet = cms.VPSet(
       cms.PSet(
           record = cms.string('CastorPedestalsRcd'),
           tag = cms.string('castor_pedestals_v1.0_test')
           ),
       cms.PSet(
           record = cms.string('CastorPedestalWidthsRcd'),
           tag = cms.string('castor_pedestalwidths_v1.0_test')
           ),
       cms.PSet(
           record = cms.string('CastorGainsRcd'),
           tag = cms.string('castor_gains_v1.0_test')
           ),
       cms.PSet(
           record = cms.string('CastorGainWidthsRcd'),
           tag = cms.string('castor_gainwidths_v1.0_test')
           ),
       cms.PSet(
           record = cms.string('CastorQIEDataRcd'),
           tag = cms.string('castor_qie_v1.0_test')
           ),
       cms.PSet(
           record = cms.string('CastorChannelQualityRcd'),
           tag = cms.string('castor_channelquality_v1.0_test')
           ),
       cms.PSet(
           record = cms.string('CastorElectronicsMapRcd'),
           tag = cms.string('castor_emap_dcc_v1.0_test')
           )
   )
)

process.castorpedestalsanalysis = cms.EDAnalyzer("CastorPedestalsAnalysis",
    hiSaveFlag  = cms.untracked.bool( False ),
    verboseflag = cms.untracked.bool( True ),
    firstTS = cms.untracked.int32(0),
    lastTS = cms.untracked.int32(9),
    castorDigiCollectionTag = cms.InputTag('castorDigis')
)


process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.enable = False
process.MessageLogger.files.detailedInfo = dict(extension = 'txt')

#process.p = cms.Path(process.dumpRaw*process.castorDigis*process.dump*process.m*process.dumpECA)
process.p = cms.Path(process.castorDigis*process.castorpedestalsanalysis)
process.ep = cms.EndPath(process.out)

