from DQM.EcalCommon.dqmpset import *
from DQM.EcalCommon.CalibCommonParams_cfi import ecalCalibCommonParams

from DQM.EcalBarrelMonitorTasks.CollectionTags_cfi import ecalDQMCollectionTags
from DQM.EcalBarrelMonitorTasks.PedestalTask_cfi import pedestalTask
from DQM.EcalBarrelMonitorTasks.TestPulseTask_cfi import testPulseTask
from DQM.EcalBarrelMonitorTasks.LaserTask_cfi import laserTask
from DQM.EcalBarrelMonitorTasks.LedTask_cfi import ledTask
from DQM.EcalBarrelMonitorTasks.PNDiodeTask_cfi import pnDiodeTask

ecalPedstalMonitorTask = cms.EDAnalyzer("EcalDQMonitorTask",
    moduleName = cms.untracked.string("EcalPedestal Monitor Source"),
    workers = cms.untracked.vstring("PedestalTask"),
    workerParameters = dqmpset(dict(PedestalTask = pedestalTask, common = ecalCalibCommonParams)),
    collectionTags = ecalDQMCollectionTags
)
ecalTestPulseMonitorTask = cms.EDAnalyzer("EcalDQMonitorTask",
    moduleName = cms.untracked.string("EcalTestPulse Monitor Source"),
    workers = cms.untracked.vstring("TestPulseTask"),
    workerParameters = dqmpset(dict(TestPulseTask = testPulseTask, common = ecalCalibCommonParams)),
    collectionTags = ecalDQMCollectionTags
)
ecalLaserLedMonitorTask = cms.EDAnalyzer("EcalDQMonitorTask",
    moduleName = cms.untracked.string("EcalLaserLed Monitor Source"),
    workers = cms.untracked.vstring("LaserTask", "LedTask"),
    workerParameters = dqmpset(dict(LaserTask = laserTask, LedTask = ledTask, common = ecalCalibCommonParams)),
    collectionTags = ecalDQMCollectionTags
)
ecalPNDiodeMonitorTask = cms.EDAnalyzer("EcalDQMonitorTask",
    moduleName = cms.untracked.string("EcalPNDiode Monitor Source"),
    workers = cms.untracked.vstring("PNDiodeTask"),
    workerParameters = dqmpset(dict(PNDiodeTask = pnDiodeTask, common = ecalCalibCommonParams)),
    collectionTags = ecalDQMCollectionTags
)
