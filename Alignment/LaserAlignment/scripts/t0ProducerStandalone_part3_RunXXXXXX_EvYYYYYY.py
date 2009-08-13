    ),
)

process.load( "EventFilter.SiStripRawToDigi.SiStripDigis_cfi" )
process.siStripDigis.ProductLabel = "source"
#process.siStripDigis.ProductLabel = "hltCalibrationRaw"

process.load( "Configuration.StandardSequences.FrontierConditions_GlobalTag_cff" )
process.GlobalTag.globaltag = 'GR09_31X_V5P::All'

process.eventFilter = cms.EDFilter("LaserAlignmentEventFilter",
    #pick Run and Event range
    RunFirst = cms.untracked.int32(XXXXXX),
    RunLast = cms.untracked.int32(XXXXXX), 
    EventFirst = cms.untracked.int32(YYYYYY),
    EventLast = cms.untracked.int32(000000)
)

# multiple sets can be given, only those will be output
process.load( "Alignment.LaserAlignment.LaserAlignmentT0Producer_cfi" )
process.laserAlignmentT0Producer.DigiProducerList = cms.VPSet(
  cms.PSet(
    DigiLabel = cms.string( 'ZeroSuppressed' ),
    DigiType = cms.string( 'Processed' ),
    DigiProducer = cms.string( 'siStripDigis' ),
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
process.LaserAlignmentT0ProducerDQM.PlainOutputFileName = cms.string( "TkAlLAS_RunXXXXXX_EvYYYYYY_F0.dqm.root" )

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32( -1 )
#  input = cms.untracked.int32( 100)
)

process.out = cms.OutputModule( "PoolOutputModule",
  outputCommands = cms.untracked.vstring(
    'drop *', 
    'keep *_laserAlignmentT0Producer_*_*'
  ),
  fileName = cms.untracked.string( 'TkAlLAS_RunXXXXXX_EvYYYYYY_F0.root' )
)

process.p = cms.Path( process.eventFilter+
                      process.siStripDigis+
                      process.laserAlignmentT0Producer+
                      process.LaserAlignmentT0ProducerDQM+
                      process.out )
