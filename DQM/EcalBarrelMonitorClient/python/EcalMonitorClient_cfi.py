import FWCore.ParameterSet.Config as cms

from DQM.EcalCommon.dqmpset import *
from DQM.EcalCommon.CommonParams_cfi import ecalCommonParams

from DQM.EcalBarrelMonitorClient.IntegrityClient_cfi import integrityClient
from DQM.EcalBarrelMonitorClient.OccupancyClient_cfi import occupancyClient
from DQM.EcalBarrelMonitorClient.PresampleClient_cfi import presampleClient
from DQM.EcalBarrelMonitorClient.TrigPrimClient_cfi import trigPrimClient
from DQM.EcalBarrelMonitorClient.RawDataClient_cfi import rawDataClient
from DQM.EcalBarrelMonitorClient.TimingClient_cfi import timingClient
from DQM.EcalBarrelMonitorClient.SelectiveReadoutClient_cfi import selectiveReadoutClient
from DQM.EcalBarrelMonitorClient.SummaryClient_cfi import summaryClient

ecalMonitorClient = cms.EDAnalyzer("EcalDQMonitorClient",
    moduleName = cms.untracked.string("Ecal Monitor Client"),
    mergeRuns = cms.untracked.bool(False),
    # workers to be turned on
    workers = cms.untracked.vstring(
        "IntegrityClient",
        "OccupancyClient",
        "PresampleClient",
        "TrigPrimClient",
        "RawDataClient",
        "TimingClient",
        "SelectiveReadoutClient",
        "SummaryClient"
    ),
    # task parameters (included from indivitual cfis)
    workerParameters = dqmpset(
        dict(
            IntegrityClient = integrityClient,
            OccupancyClient = occupancyClient,
            PresampleClient = presampleClient,
            TrigPrimClient = trigPrimClient,
            RawDataClient = rawDataClient,
            TimingClient = timingClient,
            SelectiveReadoutClient = selectiveReadoutClient,
            SummaryClient = summaryClient,
            common = ecalCommonParams
        )
    ),
    verbosity = cms.untracked.int32(0)
)
