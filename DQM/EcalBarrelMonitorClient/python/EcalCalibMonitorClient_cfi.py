import FWCore.ParameterSet.Config as cms

from DQM.EcalCommon.dqmpset import *
from DQM.EcalCommon.CalibCommonParams_cfi import ecalCalibCommonParams

from DQM.EcalBarrelMonitorClient.LaserClient_cfi import ecalLaserClient
from DQM.EcalBarrelMonitorClient.LedClient_cfi import ecalLedClient
from DQM.EcalBarrelMonitorClient.TestPulseClient_cfi import ecalTestPulseClient
from DQM.EcalBarrelMonitorClient.PedestalClient_cfi import ecalPedestalClient
from DQM.EcalBarrelMonitorClient.PNIntegrityClient_cfi import ecalPnIntegrityClient
from DQM.EcalBarrelMonitorClient.CalibrationSummaryClient_cfi import ecalCalibrationSummaryClient

ecalCalibMonitorClient = cms.EDAnalyzer("EcalDQMonitorClient",
    moduleName = cms.untracked.string("EcalCalib Monitor Client"),
    mergeRuns = cms.untracked.bool(False),
    # workers to be turned on
    workers = cms.untracked.vstring(
        "LaserClient",
        "LedClient",
        "TestPulseClient",
#        "PedestalClient",
        "PNIntegrityClient",
        "CalibrationSummaryClient"
    ),
    # task parameters (included from indivitual cfis)
    workerParameters = dqmpset(
        dict(
            LaserClient = ecalLaserClient,
            LedClient = ecalLedClient,
            TestPulseClient = ecalTestPulseClient,
#            PedestalClient = ecalPedestalClient,
            PNIntegrityClient = ecalPnIntegrityClient,
            CalibrationSummaryClient = ecalCalibrationSummaryClient,
            common = ecalCalibCommonParams
        )
    ),
    PNMaskFile = cms.untracked.FileInPath("DQM/EcalBarrelMonitorClient/data/mask-PN.txt"),
    verbosity = cms.untracked.int32(0),
    online = cms.untracked.bool(True)
)
