
import FWCore.ParameterSet.Config as cms

process = cms.Process( "laserAlignmentT0ProducerProcess" )

process.MessageLogger = cms.Service( "MessageLogger",
  cerr = cms.untracked.PSet(
    threshold = cms.untracked.string( 'ERROR' )
  ),
  cout = cms.untracked.PSet(
    threshold = cms.untracked.string( 'INFO' )
  ),
  destinations = cms.untracked.vstring( 'cout', 'cerr' )
)

process.source = cms.Source( "PoolSource",
  skipEvents = cms.untracked.uint32( 7000 ), #################################
  fileNames = cms.untracked.vstring(
#    '/store/data/Commissioning08/TestEnables/RAW/v1/000/070/664/068DFAE6-7EAF-DD11-9B0A-001617DBD288.root' #1
#    '/store/data/Commissioning08/TestEnables/RAW/v1/000/070/664/084D6BEF-84AF-DD11-9B5C-000423D991D4.root' #2
#    '/store/data/Commissioning08/TestEnables/RAW/v1/000/070/664/0ED4E41F-7BAF-DD11-8AD0-000423D6B42C.root' #3
#    '/store/data/Commissioning08/TestEnables/RAW/v1/000/070/664/24F2178A-7CAF-DD11-BAB8-001617E30F58.root' #4
#    '/store/data/Commissioning08/TestEnables/RAW/v1/000/070/664/287142EC-7BAF-DD11-84D9-001617C3B6C6.root' #5
#    '/store/data/Commissioning08/TestEnables/RAW/v1/000/070/664/2C7978EC-84AF-DD11-B849-000423D98E54.root' #6
#    '/store/data/Commissioning08/TestEnables/RAW/v1/000/070/664/3018A1C0-80AF-DD11-ACA8-001617E30CA4.root' #7
    '/store/data/Commissioning08/TestEnables/RAW/v1/000/070/664/3E304C78-7AAF-DD11-BD61-000423D98FBC.root' #8
#    '/store/data/Commissioning08/TestEnables/RAW/v1/000/070/664/4C4F4870-81AF-DD11-ABF5-000423DD2F34.root' #9
#    '/store/data/Commissioning08/TestEnables/RAW/v1/000/070/664/54AB9B26-82AF-DD11-8226-000423D99F3E.root' #10
#    '/store/data/Commissioning08/TestEnables/RAW/v1/000/070/664/54D61709-80AF-DD11-9EB3-0016177CA7A0.root' #11
#    '/store/data/Commissioning08/TestEnables/RAW/v1/000/070/664/72514E86-83AF-DD11-95BE-001617DBD316.root' #12
#    '/store/data/Commissioning08/TestEnables/RAW/v1/000/070/664/763E1498-83AF-DD11-B29E-001617DC1F70.root' #13
#    '/store/data/Commissioning08/TestEnables/RAW/v1/000/070/664/86EBF63A-7DAF-DD11-9887-001617C3B6DE.root' #14
#    '/store/data/Commissioning08/TestEnables/RAW/v1/000/070/664/A031B957-7FAF-DD11-BB48-001617C3B5D8.root' #15
#    '/store/data/Commissioning08/TestEnables/RAW/v1/000/070/664/B2ADC13B-7DAF-DD11-84CE-001617C3B5F4.root' #16
#    '/store/data/Commissioning08/TestEnables/RAW/v1/000/070/664/D68DA36E-9BAF-DD11-9804-001D09F24448.root' #17
#    '/store/data/Commissioning08/TestEnables/RAW/v1/000/070/664/DA2534F1-7DAF-DD11-82A5-001617C3B5D8.root' #18
  )
)




process.load( "EventFilter.SiStripRawToDigi.SiStripDigis_cfi" )
process.siStripDigis.ProductLabel = "hltCalibrationRaw"

#process.load( "CondCore.DBCommon.CondDBSetup_cfi" )
#process.allSource = cms.ESSource( "PoolDBESSource",
#  process.CondDBSetup,
#  connect = cms.string( 'frontier://FrontierProd/CMS_COND_20X_GLOBALTAG' ),
#  globaltag = cms.string( 'IDEAL_v2::All' )                                     
#)
process.load( "Configuration.StandardSequences.FrontierConditions_GlobalTag_cff" )
process.GlobalTag.globaltag = 'CRAFT_V4P::All'



# multiple sets can be given, only those will be output
process.load( "Alignment.LaserAlignment.LaserAlignmentT0Producer_cfi" )
process.laserAlignmentT0Producer.DigiProducerList = cms.VPSet(
  cms.PSet(
    DigiLabel = cms.string( 'ZeroSuppressed' ),
    DigiType = cms.string( 'Processed' ),
    DigiProducer = cms.string( 'siStripDigis' )
  )
)

process.load( "DQMServices.Core.DQM_cfg" )
process.load( "DQMOffline.Alignment.LaserAlignmentT0ProducerDQM_cfi" )
process.LaserAlignmentT0ProducerDQM.DigiProducerList = cms.VPSet(
  cms.PSet(
    DigiLabel = cms.string( 'ZeroSuppressed' ),
    DigiType = cms.string( 'Processed' ),
    DigiProducer = cms.string( 'siStripDigis' )
  )
)
process.LaserAlignmentT0ProducerDQM.OutputInPlainROOT = True;
process.LaserAlignmentT0ProducerDQM.UpperAdcThreshold = cms.uint32( 280 )

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32( 10000 )
)

process.out = cms.OutputModule( "PoolOutputModule",
  outputCommands = cms.untracked.vstring(
    'drop *', 
    'keep *_laserAlignmentT0Producer_*_*'
  ),
  fileName = cms.untracked.string( 'TkAlLAS.root' )
)




process.seqDigitization = cms.Path( process.siStripDigis )

#process.dump  = cms.EDAnalyzer("EventContentAnalyzer")
#process.dumpP = cms.Path( process.dump )

process.seqAnalysis = cms.Path( process.laserAlignmentT0Producer + process.LaserAlignmentT0ProducerDQM )
process.outputPath = cms.EndPath( process.out )



