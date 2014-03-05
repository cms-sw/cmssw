import FWCore.ParameterSet.Config as cms

from DQM.EcalCommon.CommonParams_cfi import ecalCommonParams

from DQM.EcalBarrelMonitorTasks.CollectionTags_cfi import ecalDQMCollectionTags
from DQM.EcalBarrelMonitorTasks.PedestalTask_cfi import ecalPedestalTask
from DQM.EcalBarrelMonitorTasks.TestPulseTask_cfi import ecalTestPulseTask
from DQM.EcalBarrelMonitorTasks.LaserTask_cfi import ecalLaserTask
from DQM.EcalBarrelMonitorTasks.LedTask_cfi import ecalLedTask
from DQM.EcalBarrelMonitorTasks.PNDiodeTask_cfi import ecalPNDiodeTask

ecalPedestalMonitorTask = cms.EDAnalyzer("EcalDQMonitorTask",
    moduleName = cms.untracked.string("EcalPedestal Monitor Source"),
    workers = cms.untracked.vstring("PedestalTask"),
    workerParameters = cms.untracked.PSet(
        PedestalTask = ecalPedestalTask
    ),
    commonParameters = ecalCommonParams,
    collectionTags = ecalDQMCollectionTags,
    resetInterval = cms.untracked.double(2.),
    verbosity = cms.untracked.int32(0)
)
ecalTestPulseMonitorTask = cms.EDAnalyzer("EcalDQMonitorTask",
    moduleName = cms.untracked.string("EcalTestPulse Monitor Source"),
    workers = cms.untracked.vstring("TestPulseTask"),
    workerParameters = cms.untracked.PSet(
        TestPulseTask = ecalTestPulseTask
    ),
    commonParameters = ecalCommonParams,
    collectionTags = ecalDQMCollectionTags,
    resetInterval = cms.untracked.double(2.),
    verbosity = cms.untracked.int32(0)                                          
)
ecalLaserLedMonitorTask = cms.EDAnalyzer("EcalDQMonitorTask",
    moduleName = cms.untracked.string("EcalLaserLed Monitor Source"),
    workers = cms.untracked.vstring("LaserTask", "LedTask"),
    workerParameters = cms.untracked.PSet(
        LaserTask = ecalLaserTask,
        LedTask = ecalLedTask
    ),
    commonParameters = ecalCommonParams,
    collectionTags = ecalDQMCollectionTags,
    resetInterval = cms.untracked.double(2.),
    verbosity = cms.untracked.int32(0)                                         
)
ecalPNDiodeMonitorTask = cms.EDAnalyzer("EcalDQMonitorTask",
    moduleName = cms.untracked.string("EcalPNDiode Monitor Source"),
    workers = cms.untracked.vstring("PNDiodeTask"),
    workerParameters = cms.untracked.PSet(
        PNDiodeTask = ecalPNDiodeTask
    ),
    commonParameters = ecalCommonParams,
    collectionTags = ecalDQMCollectionTags,
    resetInterval = cms.untracked.double(2.),
    verbosity = cms.untracked.int32(0)
)
