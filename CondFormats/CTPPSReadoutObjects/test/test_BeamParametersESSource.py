import FWCore.ParameterSet.Config as cms

process = cms.Process('test')

# minimum of logs
process.MessageLogger = cms.Service("MessageLogger",
  statistics = cms.untracked.vstring(),
  destinations = cms.untracked.vstring('cout'),
  cout = cms.untracked.PSet(
    threshold = cms.untracked.string('INFO')
  )
)

# raw data source
#process.source = cms.Source("EmptySource",
#)

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(1)
)

# A data source must always be defined. We don't need it, so here's a dummy one.
process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    # timetype = cms.string('lumiid'),
    firstValue = cms.uint64(1),
    lastValue = cms.uint64(1),
    interval = cms.uint64(1)
)

# Load beam parameters
process.load("CondFormats.CTPPSReadoutObjects.CTPPSBeamParametersESSource_cfi")
# process.CTPPSBeamParametersESSource.verbosity = cms.untracked.uint32(2)

# Test retrieving the ESSource data 
process.get = cms.EDAnalyzer("RetrieveCTPPSBeamParameters",
    verbose = cms.untracked.bool(True)
)
  
# ========================  Test writing to DB  ==========================================

# Load CondDB service
process.load("CondCore.CondDB.CondDB_cfi")

# Output database (in this case local sqlite file)
process.CondDB.connect = 'sqlite_file:CTPPSBeamParameters.db'

# We define the output service.
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDB,
    timetype = cms.untracked.string('runnumber'),
    # timetype = cms.untracked.string('timestamp'),
    # timetype = cms.untracked.string('lumiid'),
    toPut = cms.VPSet(
      cms.PSet(
          record = cms.string('CTPPSBeamParametersRcd'),
          tag = cms.string('CTPPSBeamParameters_v1')
      )
    )
)

process.write = cms.EDAnalyzer("WriteCTPPSBeamParameters",
    record = cms.string('CTPPSBeamParametersRcd'),
    loggingOn= cms.untracked.bool(True),
    SinceAppendMode=cms.bool(True),
    Source=cms.PSet(
        IOVRun=cms.untracked.uint32(1)
    )
)

# ===============================================================================


process.path = cms.Path(
  process.get*process.write
)
