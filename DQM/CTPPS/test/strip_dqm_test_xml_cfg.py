# This config can be used for tests of XML files containing mappings.
# Since data in CondDB has same labels ESPrefer is needed.
# For internal and testing purposes only

import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Run3_cff import Run3

process = cms.Process('RECODQM', Run3)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10000) )
process.verbosity = cms.untracked.PSet( input = cms.untracked.int32(-1) )

# minimum of logs
process.MessageLogger = cms.Service("MessageLogger",
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

# data source
process.source = cms.Source("NewEventStreamFileReader",
  fileNames = cms.untracked.vstring(
         'file:/eos/cms/store/t0streamer/Data/PhysicsZeroBias2/000/369/585/run369585_ls0044_streamPhysicsZeroBias2_StorageManager.dat'
  ),
)

from Configuration.AlCa.GlobalTag import GlobalTag
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag = GlobalTag(process.GlobalTag, autoCond['run3_data_prompt'], '')

# raw-to-digi conversion
process.load("EventFilter.CTPPSRawToDigi.ctppsRawToDigi_xml_cff")

# prefer mappings from XML files
process.es_prefer_totemTimingMapping = cms.ESPrefer("TotemDAQMappingESSourceXML", "totemDAQMappingESSourceXML_TotemTiming", TotemReadoutRcd=cms.vstring("TotemDAQMapping/TotemTiming"))
process.es_prefer_totemDiamondMapping = cms.ESPrefer("TotemDAQMappingESSourceXML", "totemDAQMappingESSourceXML_TimingDiamond", TotemReadoutRcd=cms.vstring("TotemDAQMapping/TimingDiamond"))
process.es_prefer_totemT2Mapping = cms.ESPrefer("TotemDAQMappingESSourceXML", "totemDAQMappingESSourceXML_TotemT2", TotemReadoutRcd=cms.vstring("TotemDAQMapping/TotemT2"))
process.es_prefer_TrackingStripMapping = cms.ESPrefer("TotemDAQMappingESSourceXML", "totemDAQMappingESSourceXML_TrackingStrip", TotemReadoutRcd=cms.vstring("TotemDAQMapping/TrackingStrip"))

# local RP reconstruction chain with standard settings
process.load("RecoPPS.Configuration.recoCTPPS_cff")
process.load('Geometry.VeryForwardGeometry.geometryRPFromDD_2021_cfi')
# CTPPS DQM modules
process.load("DQM.CTPPS.ctppsDQM_cff")
process.ctppsDiamondDQMSource.excludeMultipleHits = cms.bool(True)
process.ctppsDiamondDQMSource.plotOnline = cms.untracked.bool(True)
process.ctppsDiamondDQMSource.plotOffline = cms.untracked.bool(False)
process.path = cms.Path(
    process.ctppsRawToDigi*
    process.recoCTPPS*
    process.ctppsDQMCalibrationSource*
    process.ctppsDQMCalibrationHarvest
)

process.end_path = cms.EndPath(
    process.dqmEnv +
    process.dqmSaver
)

process.schedule = cms.Schedule(
    process.path,
    process.end_path
)
