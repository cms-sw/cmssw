import FWCore.ParameterSet.Config as cms
import string

process = cms.Process('RECODQM')

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000) )
process.verbosity = cms.untracked.PSet( input = cms.untracked.int32(-1) )

# minimum of logs
process.MessageLogger = cms.Service("MessageLogger",
    statistics = cms.untracked.vstring(),
    destinations = cms.untracked.vstring('cerr'),
    cerr = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING')
    )
)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

# load DQM framework
process.load("DQM.Integration.config.environment_cfi")
process.dqmEnv.subSystemFolder = "CTPPS"
process.dqmEnv.eventInfoFolder = "EventInfo"
process.dqmSaver.path = ""
process.dqmSaver.tag = "CTPPS"

process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring( *(
    # '''/store/data/Commissioning2018/ZeroBias/RAW/v1/000/314/816/00000/0062E91C-2245-E811-8DCB-FA163EE59A93.root',
    # '''/store/data/Commissioning2018/ZeroBias/RAW/v1/000/314/816/00000/007EABA4-5745-E811-8573-FA163EB3E1C0.root',
# '''/store/data/Commissioning2018/MinimumBias/RAW/v1/000/314/276/00000/B45F5174-6040-E811-BCDA-FA163EB38F53.root',
'/store/data/Commissioning2018/ZeroBias1/RAW/v1/000/314/277/00000/04F64F14-A53F-E811-A60D-FA163ED486E3.root',
    )
    )
)


from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_hlt_relval', '')

# raw-to-digi conversion
process.load("EventFilter.CTPPSRawToDigi.ctppsRawToDigi_cff")

# local RP reconstruction chain with standard settings
process.load("RecoCTPPS.Configuration.recoCTPPS_cff")

# load local geometry to avoid GT
process.load('Geometry.VeryForwardGeometry.geometryRPFromDD_2018_cfi')
process.load('RecoCTPPS.TotemRPLocal.totemTimingLocalReconstruction_cff')

# CTPPS DQM modules
process.load("DQM.CTPPS.ctppsDQM_cff")

process.path = cms.Path(
    process.ctppsRawToDigi *
    process.recoCTPPS *
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
