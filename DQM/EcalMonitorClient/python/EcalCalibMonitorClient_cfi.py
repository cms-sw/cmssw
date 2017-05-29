import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

from DQM.EcalCommon.CommonParams_cfi import ecalCommonParams

from DQM.EcalMonitorClient.IntegrityClient_cfi import ecalIntegrityClient
from DQM.EcalMonitorClient.RawDataClient_cfi import ecalRawDataClient
from DQM.EcalMonitorClient.LaserClient_cfi import ecalLaserClient
from DQM.EcalMonitorClient.LedClient_cfi import ecalLedClient
from DQM.EcalMonitorClient.TestPulseClient_cfi import ecalTestPulseClient
from DQM.EcalMonitorClient.PedestalClient_cfi import ecalPedestalClient
from DQM.EcalMonitorClient.PNIntegrityClient_cfi import ecalPNIntegrityClient
from DQM.EcalMonitorClient.SummaryClient_cfi import ecalSummaryClient
from DQM.EcalMonitorClient.CalibrationSummaryClient_cfi import ecalCalibrationSummaryClient

ecalCalibMonitorClient = DQMEDHarvester("EcalDQMonitorClient",
    moduleName = cms.untracked.string("EcalCalib Monitor Client"),
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
    workerParameters = cms.untracked.PSet(
        IntegrityClient = ecalIntegrityClient,
        RawDataClient = ecalRawDataClient,
        LaserClient = ecalLaserClient,
        LedClient = ecalLedClient,
        TestPulseClient = ecalTestPulseClient,
        PedestalClient = ecalPedestalClient,
        PNIntegrityClient = ecalPNIntegrityClient,
        SummaryClient = ecalSummaryClient,
        CalibrationSummaryClient = ecalCalibrationSummaryClient
    ),
    commonParameters = ecalCommonParams,
#    PNMaskFile = cms.untracked.FileInPath("DQM/EcalMonitorClient/data/mask-PN.txt"),
    verbosity = cms.untracked.int32(0)
)

