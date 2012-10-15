import FWCore.ParameterSet.Config as cms

from DQM.EcalCommon.CalibCommonParams_cfi import ecalCalibCommonParams

from DQM.EcalBarrelMonitorTasks.CollectionTags_cfi import ecalDQMCollectionTags
from DQM.EcalBarrelMonitorTasks.PedestalTask_cfi import ecalPedestalTask
from DQM.EcalBarrelMonitorTasks.TestPulseTask_cfi import ecalTestPulseTask
from DQM.EcalBarrelMonitorTasks.LaserTask_cfi import ecalLaserTask
from DQM.EcalBarrelMonitorTasks.LedTask_cfi import ecalLedTask
from DQM.EcalBarrelMonitorTasks.PNDiodeTask_cfi import ecalPnDiodeTask

ecalPedestalMonitorTask = cms.EDAnalyzer("EcalDQMonitorTask",
    moduleName = cms.untracked.string("EcalPedestal Monitor Source"),
    workers = cms.untracked.vstring("PedestalTask"),
    workerParameters = cms.untracked.PSet(
        PedestalTask = ecalPedestalTask,
        common = ecalCalibCommonParams
    ),
    collectionTags = ecalDQMCollectionTags,
    online = cms.untracked.bool(True),
    resetInterval = cms.untracked.double(2.),
    verbosity = cms.untracked.int32(0)
)
ecalTestPulseMonitorTask = cms.EDAnalyzer("EcalDQMonitorTask",
    moduleName = cms.untracked.string("EcalTestPulse Monitor Source"),
    workers = cms.untracked.vstring("TestPulseTask"),
    workerParameters = cms.untracked.PSet(
        TestPulseTask = ecalTestPulseTask,
        common = ecalCalibCommonParams
    ),
    collectionTags = ecalDQMCollectionTags,
    online = cms.untracked.bool(True),
    resetInterval = cms.untracked.double(2.),
    verbosity = cms.untracked.int32(0)                                          
)
ecalLaserLedMonitorTask = cms.EDAnalyzer("EcalDQMonitorTask",
    moduleName = cms.untracked.string("EcalLaserLed Monitor Source"),
    workers = cms.untracked.vstring("LaserTask", "LedTask"),
    workerParameters = cms.untracked.PSet(
        LaserTask = ecalLaserTask,
        LedTask = ecalLedTask,
        common = ecalCalibCommonParams
    ),
    collectionTags = ecalDQMCollectionTags,
    online = cms.untracked.bool(True),
    resetInterval = cms.untracked.double(2.),
    verbosity = cms.untracked.int32(0)                                         
)
ecalPNDiodeMonitorTask = cms.EDAnalyzer("EcalDQMonitorTask",
    moduleName = cms.untracked.string("EcalPNDiode Monitor Source"),
    workers = cms.untracked.vstring("PNDiodeTask"),
    workerParameters = cms.untracked.PSet(
        PNDiodeTask = ecalPnDiodeTask,
        common = ecalCalibCommonParams
    ),
    collectionTags = ecalDQMCollectionTags,
    online = cms.untracked.bool(True),
    resetInterval = cms.untracked.double(2.),
    verbosity = cms.untracked.int32(0)
)
