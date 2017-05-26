import FWCore.ParameterSet.Config as cms
import string

process = cms.Process('RECODQM')

process.maxEvents = cms.untracked.PSet(
input = cms.untracked.int32(-1)
)

process.verbosity = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# diamonds mapping
process.totemDAQMappingESSourceXML_TimingDiamond = cms.ESSource("TotemDAQMappingESSourceXML",
  verbosity = cms.untracked.uint32(0),
  subSystem = cms.untracked.string("TimingDiamond"),
  configuration = cms.VPSet(
    # before diamonds inserted in DAQ
    cms.PSet(
      validityRange = cms.EventRange("1:min - 283819:max"),
      mappingFileNames = cms.vstring(),
      maskFileNames = cms.vstring()
    ),
    # after diamonds inserted in DAQ
    cms.PSet(
      validityRange = cms.EventRange("283820:min - 999999999:max"),
      mappingFileNames = cms.vstring("CondFormats/CTPPSReadoutObjects/xml/mapping_timing_diamond.xml"),
      maskFileNames = cms.vstring()
    )
  )
)

# minimum of logs
process.MessageLogger = cms.Service("MessageLogger",
    statistics = cms.untracked.vstring(),
    destinations = cms.untracked.vstring('cerr'),
    cerr = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING')
    )
)

# load DQM framework
process.load("DQM.Integration.config.environment_cfi")
process.dqmEnv.subSystemFolder = "CTPPS"
process.dqmEnv.eventInfoFolder = "EventInfo"
process.dqmSaver.path = ""
process.dqmSaver.tag = "CTPPS"


# raw data source
process.source = cms.Source("NewEventStreamFileReader",
  fileNames = cms.untracked.vstring(
    #'file:/afs/cern.ch/user/j/jkaspar/public/run273062_ls0001-2_stream.root'
        '/store/t0streamer/Data/Physics/000/294/737/run294737_ls0011_streamPhysics_StorageManager.dat',
        '/store/t0streamer/Data/Physics/000/294/737/run294737_ls0012_streamPhysics_StorageManager.dat',
        '/store/t0streamer/Data/Physics/000/294/737/run294737_ls0013_streamPhysics_StorageManager.dat',
        '/store/t0streamer/Data/Physics/000/294/737/run294737_ls0014_streamPhysics_StorageManager.dat',
        '/store/t0streamer/Data/Physics/000/294/737/run294737_ls0015_streamPhysics_StorageManager.dat',
        '/store/t0streamer/Data/Physics/000/294/737/run294737_ls0016_streamPhysics_StorageManager.dat',
        '/store/t0streamer/Data/Physics/000/294/737/run294737_ls0017_streamPhysics_StorageManager.dat',
        '/store/t0streamer/Data/Physics/000/294/737/run294737_ls0018_streamPhysics_StorageManager.dat',
        '/store/t0streamer/Data/Physics/000/294/737/run294737_ls0019_streamPhysics_StorageManager.dat',
        '/store/t0streamer/Data/Physics/000/294/737/run294737_ls0020_streamPhysics_StorageManager.dat',
        '/store/t0streamer/Data/Physics/000/294/737/run294737_ls0021_streamPhysics_StorageManager.dat',
        '/store/t0streamer/Data/Physics/000/294/737/run294737_ls0022_streamPhysics_StorageManager.dat',
        '/store/t0streamer/Data/Physics/000/294/737/run294737_ls0023_streamPhysics_StorageManager.dat',
        '/store/t0streamer/Data/Physics/000/294/737/run294737_ls0024_streamPhysics_StorageManager.dat',
        '/store/t0streamer/Data/Physics/000/294/737/run294737_ls0025_streamPhysics_StorageManager.dat',
        '/store/t0streamer/Data/Physics/000/294/737/run294737_ls0026_streamPhysics_StorageManager.dat',
        '/store/t0streamer/Data/Physics/000/294/737/run294737_ls0027_streamPhysics_StorageManager.dat',
        '/store/t0streamer/Data/Physics/000/294/737/run294737_ls0028_streamPhysics_StorageManager.dat',
        '/store/t0streamer/Data/Physics/000/294/737/run294737_ls0029_streamPhysics_StorageManager.dat',
        '/store/t0streamer/Data/Physics/000/294/737/run294737_ls0030_streamPhysics_StorageManager.dat',
  )
)


# raw-to-digi conversion
process.load("EventFilter.CTPPSRawToDigi.ctppsRawToDigi_cff")

# local RP reconstruction chain with standard settings
process.load("RecoCTPPS.Configuration.recoCTPPS_cff")

# rechits production
process.load('Geometry.VeryForwardGeometry.geometryRP_cfi')
process.load('RecoCTPPS.TotemRPLocal.ctppsDiamondRecHits_cfi')

# local tracks fitter
process.load('RecoCTPPS.TotemRPLocal.ctppsDiamondLocalTracks_cfi')


# CTPPS DQM modules
process.load("DQM.CTPPS.ctppsDQM_cff")
process.ctppsDiamondDQMSource.excludeMultipleHits = cms.bool(True);

process.path = cms.Path(
  process.ctppsRawToDigi *
  process.recoCTPPS *
  process.ctppsDiamondRawToDigi *
  process.ctppsDiamondRecHits *
  process.ctppsDiamondLocalTracks *
  process.ctppsDQM 
)

process.end_path = cms.EndPath(
  process.dqmEnv +
  process.dqmSaver
)

process.schedule = cms.Schedule(
  process.path,
  process.end_path
)
