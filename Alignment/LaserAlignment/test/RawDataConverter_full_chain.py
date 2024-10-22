import FWCore.ParameterSet.Config as cms

process = cms.Process( "RawDataConverter" )

## Choose the input files (and if needed the Event ranges)
process.source = cms.Source( "PoolSource",
  fileNames = cms.untracked.vstring(
       '/store/data/Run2010A/TestEnables/RAW/v1/000/140/124/56E00D1B-308F-DF11-BE54-001D09F24691.root'              
    ),
    eventsToProcess = cms.untracked.VEventRange(
    '140124:1141118271-140124:2134733046'
    )
)

# Choose how many events should be processed
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32( 2000 )
)

## message logger
process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        threshold = cms.untracked.string('ERROR')
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('INFO')
    ),
    debugModules = cms.untracked.vstring('RawDataConverter')
)


# Choose the correct global tag for the data
process.load( "Configuration.StandardSequences.FrontierConditions_GlobalTag_cff" )
process.GlobalTag.globaltag = cms.string('GR_R_37X_V6::All')


# Laser Alignment Event Filter
process.load('Alignment.LaserAlignment.LaserAlignmentEventFilter_cfi')


# strip digitizer
process.load( "EventFilter.SiStripRawToDigi.SiStripDigis_cfi" )
process.siStripDigis.ProductLabel = 'source'
process.siStripDigis.TriggerFedId = -1


# multiple sets can be given, only those will be output
process.load( "Alignment.LaserAlignment.LaserAlignmentT0Producer_cfi" )
process.laserAlignmentT0Producer.DigiProducerList = cms.VPSet(
  cms.PSet(
    DigiLabel = cms.string( 'ZeroSuppressed' ),
    DigiType = cms.string( 'Processed' ),
    DigiProducer = cms.string( 'siStripDigis' )
  )
)


# Raw Data Converter
process.load('Alignment.LaserAlignment.RawDataConverter_cfi')
process.RawDataConverter.OutputFileName = cms.untracked.string( 'RawDataConverter.root' ) #@@@ output file


# Define what should go into the Producer Output file (in this case we choose only th T0 Producer products)
process.out = cms.OutputModule( "PoolOutputModule",
  outputCommands = cms.untracked.vstring(
    'drop *', 
    'keep *_laserAlignmentT0Producer_*_*'
  ),
  fileName = cms.untracked.string( '/tmp/wittmer/RawDataConverterTest_T0.root' )
)

# Run the full chain of LAS analysis
process.p = cms.Path(process.LaserAlignmentEventFilter + process.siStripDigis + process.laserAlignmentT0Producer + process.RawDataConverter + process.out)
