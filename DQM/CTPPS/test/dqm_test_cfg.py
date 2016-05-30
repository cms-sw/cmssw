import FWCore.ParameterSet.Config as cms

process = cms.Process('RECODQM')

# minimum of logs
process.MessageLogger = cms.Service("MessageLogger",
    statistics = cms.untracked.vstring(),
    destinations = cms.untracked.vstring('cerr'),
    cerr = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING')
    )
)

# global tag
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:mc', '')  #for MC

# load DQM frame work
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.load("DQMServices.Components.DQMStoreStats_cfi")

process.dqmEnv.subSystemFolder = 'CTPPS'

process.dqmSaver = cms.EDAnalyzer("DQMFileSaverOnline",
  producer = cms.untracked.string("DQM"),
  tag = cms.untracked.string("CTPPS"),
  path = cms.untracked.string("."),
)

# RP raw data
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:/afs/cern.ch/user/j/jkaspar/public/run273062_ls0001-2_stream.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# RP digi
process.load('CondFormats.TotemReadoutObjects.TotemDAQMappingESSourceXML_cfi')
process.TotemDAQMappingESSourceXML.mappingFileNames.append("CondFormats/TotemReadoutObjects/xml/ctpps_210_mapping.xml")

process.load("EventFilter.TotemRawToDigi.totemTriggerRawToDigi_cfi")
process.totemTriggerRawToDigi.rawDataTag = cms.InputTag("rawDataCollector")

process.load('EventFilter.TotemRawToDigi.totemRPRawToDigi_cfi')
process.totemRPRawToDigi.rawDataTag = cms.InputTag("rawDataCollector")

# RP geometry
process.load("Geometry.VeryForwardGeometry.geometryRP_cfi")
process.XMLIdealGeometryESSource.geomXMLFiles.append("Geometry/VeryForwardData/data/RP_Garage/RP_Dist_Beam_Cent.xml")

# local RP reconstruction chain with standard settings
process.load("RecoCTPPS.TotemRPLocal.totemRPLocalReconstruction_cff")

# TOTEM DQM modules
process.load("DQM.CTPPS.totemDAQTriggerDQMSource_cfi")
process.load("DQM.CTPPS.totemRPDQMSource_cfi")

# execution schedule
process.reco_totem = cms.Path(
  process.totemTriggerRawToDigi *
  process.totemRPRawToDigi *
  process.totemRPLocalReconstruction
)

process.dqm_totem = cms.Path(
  process.totemDAQTriggerDQMSource *
  process.totemRPDQMSource
)

process.dqm_common = cms.Path(
    process.dqmEnv *
    process.dqmSaver
    #process.dqmStoreStats
)

process.schedule = cms.Schedule(
    process.reco_totem,
    process.dqm_totem,
    process.dqm_common
)
