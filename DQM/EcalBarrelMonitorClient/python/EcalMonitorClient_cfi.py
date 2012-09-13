import FWCore.ParameterSet.Config as cms

from DQM.EcalCommon.dqmpset import *
from DQM.EcalCommon.CommonParams_cfi import ecalCommonParams

from DQM.EcalBarrelMonitorClient.IntegrityClient_cfi import ecalIntegrityClient
from DQM.EcalBarrelMonitorClient.OccupancyClient_cfi import ecalOccupancyClient
from DQM.EcalBarrelMonitorClient.PresampleClient_cfi import ecalPresampleClient
from DQM.EcalBarrelMonitorClient.TrigPrimClient_cfi import ecalTrigPrimClient
from DQM.EcalBarrelMonitorClient.RawDataClient_cfi import ecalRawDataClient
from DQM.EcalBarrelMonitorClient.TimingClient_cfi import ecalTimingClient
from DQM.EcalBarrelMonitorClient.SelectiveReadoutClient_cfi import ecalSelectiveReadoutClient
from DQM.EcalBarrelMonitorClient.SummaryClient_cfi import ecalSummaryClient

ecalMonitorClient = cms.EDAnalyzer("EcalDQMonitorClient",
    moduleName = cms.untracked.string("Ecal Monitor Client"),
    mergeRuns = cms.untracked.bool(False),
    # workers to be turned on
    workers = cms.untracked.vstring(
        "IntegrityClient",
        "OccupancyClient",
        "PresampleClient",
        "RawDataClient",
        "TimingClient",
        "SummaryClient"
    ),
    # task parameters (included from indivitual cfis)
    workerParameters = dqmpset(
        dict(
            IntegrityClient = ecalIntegrityClient,
            OccupancyClient = ecalOccupancyClient,
            PresampleClient = ecalPresampleClient,
            RawDataClient = ecalRawDataClient,
            TimingClient = ecalTimingClient,
            SummaryClient = ecalSummaryClient,
            common = ecalCommonParams
        )
    ),
    verbosity = cms.untracked.int32(0)
)
