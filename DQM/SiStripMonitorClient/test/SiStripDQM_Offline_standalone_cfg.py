
import FWCore.ParameterSet.Config as cms

process = cms.Process("SiStrpDQMQTestTuning")


## Empty Event Source
process.source = cms.Source("EmptyIOVSource",
                              lastRun = cms.untracked.uint32(100),
                              timetype = cms.string('runnumber'),
                              firstValue= cms.uint64(1),
                              lastValue= cms.uint64(1),
                              interval = cms.uint64(1)
                            )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "GR09_P_V2::All"

# DQM Environment
process.load("DQMServices.Core.DQM_cfg")

process.load("Configuration.Geometry.GeometryExtended2017_cff")
process.load("CalibTracker.SiStripCommon.TkDetMapESProducer_cfi")
process.load("Geometry.TrackerGeometryBuilder.trackerParameters_cfi")
process.trackerTopology = cms.ESProducer("TrackerTopologyEP")

# SiStrip Offline DQM Client
# SiStrip Offline DQM Client
process.siStripOfflineAnalyser = cms.EDProducer("SiStripOfflineDQM",
       GlobalStatusFilling      = cms.untracked.int32(-1),
       CreateSummary            = cms.untracked.bool(False),
       SummaryConfigPath        = cms.untracked.string("DQM/SiStripMonitorClient/data/sistrip_monitorelement_config.xml"),
       UsedWithEDMtoMEConverter = cms.untracked.bool(False),
       PrintFaultyModuleList    = cms.untracked.bool(False),
       InputFileName            = cms.untracked.string("file:/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/data/Online//110/998/DQM_V0002_R000110998.root"),
       OutputFileName           = cms.untracked.string("test.root"),
       CreateTkMap              = cms.untracked.bool(True),
       TkmapParameters          = cms.untracked.PSet(
          loadFedCabling    = cms.untracked.bool(True),
          trackerdatPath    = cms.untracked.string('CommonTools/TrackerMap/data/'),
          trackermaptxtPath = cms.untracked.string('CommonTools/TrackerMap/data/'),
       ),
       TkMapOptions         = cms.untracked.vstring('QTestAlarm','FractionOfBadChannels','NumberOfCluster','NumberOfDigi','NumberOfOfffTrackCluster','NumberOfOnTrackCluster','StoNCorrOnTrack')
)


# QTest module
from DQMServices.Core.DQMQualityTester import DQMQualityTester
process.siStripQTester = DQMQualityTester(
                              qtList = cms.untracked.FileInPath('DQM/SiStripMonitorClient/data/sistrip_qualitytest_config.xml'),
                              prescaleFactor = cms.untracked.int32(1),
                              getQualityTestsFromFile = cms.untracked.bool(True)
                          )


# Tracer service
process.Tracer = cms.Service('Tracer',indentation = cms.untracked.string('$$'))
process.load('DQM.SiStripCommon.MessageLogger_cfi')

process.p1 = cms.Path(process.siStripOfflineAnalyser)
