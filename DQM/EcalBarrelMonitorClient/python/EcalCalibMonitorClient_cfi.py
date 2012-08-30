import FWCore.ParameterSet.Config as cms

from DQM.EcalCommon.dqmpset import *
from DQM.EcalCommon.CalibCommonParams_cfi import ecalCalibCommonParams

from DQM.EcalBarrelMonitorClient.LaserClient_cfi import laserClient
from DQM.EcalBarrelMonitorClient.LedClient_cfi import ledClient
from DQM.EcalBarrelMonitorClient.TestPulseClient_cfi import testPulseClient
from DQM.EcalBarrelMonitorClient.PedestalClient_cfi import pedestalClient
from DQM.EcalBarrelMonitorClient.PNIntegrityClient_cfi import pnIntegrityClient
from DQM.EcalBarrelMonitorClient.CalibrationSummaryClient_cfi import calibrationSummaryClient

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
            LaserClient = laserClient,
            LedClient = ledClient,
            TestPulseClient = testPulseClient,
#            PedestalClient = pedestalClient,
            PNIntegrityClient = pnIntegrityClient,
            CalibrationSummaryClient = calibrationSummaryClient,
            common = ecalCalibCommonParams
        )
    ),
    verbosity = cms.untracked.int32(0)
)
