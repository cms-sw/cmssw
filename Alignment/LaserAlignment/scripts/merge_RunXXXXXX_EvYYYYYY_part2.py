    ),
)

process.load( "Configuration.StandardSequences.FrontierConditions_GlobalTag_cff" )
process.GlobalTag.globaltag = 'GR09_31X_V5P::All'

process.eventFilter = cms.EDFilter("LaserAlignmentEventFilter",
    #pick Run and Event range
    RunFirst = cms.untracked.int32(XXXXXX),
    RunLast = cms.untracked.int32(XXXXXX), 
    EventFirst = cms.untracked.int32(YYYYYY),
    EventLast = cms.untracked.int32(000000)
)

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32( -1)
#  input = cms.untracked.int32( 10000)
)

process.out = cms.OutputModule( "PoolOutputModule",
  outputCommands = cms.untracked.vstring(
    'drop *', 
    'keep *_laserAlignmentT0Producer_*_*'
  ),
  fileName = cms.untracked.string( '/tmp/aperiean/RunXXXXXX/TkAlLAS_RunXXXXXX_EvYYYYYY.root')
)

process.p = cms.Path( process.eventFilter+process.out)
