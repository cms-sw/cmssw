import FWCore.ParameterSet.Config as cms

from DQM.EcalCommon.dqmpset import *
from DQM.EcalCommon.CommonParams_cfi import *

import DQM.EcalBarrelMonitorClient.IntegrityClient_cfi as ecalIntegrityClient
import DQM.EcalBarrelMonitorClient.OccupancyClient_cfi as ecalOccupancyClient
import DQM.EcalBarrelMonitorClient.PresampleClient_cfi as ecalPresampleClient
import DQM.EcalBarrelMonitorClient.TrigPrimClient_cfi as ecalTrigPrimClient
import DQM.EcalBarrelMonitorClient.RawDataClient_cfi as ecalRawDataClient
import DQM.EcalBarrelMonitorClient.TimingClient_cfi as ecalTimingClient
import DQM.EcalBarrelMonitorClient.SelectiveReadoutClient_cfi as ecalSelectiveReadoutClient
import DQM.EcalBarrelMonitorClient.SummaryClient_cfi as ecalSummaryClient

ecalMonitorClientPaths = dict(
    IntegrityClient = ecalIntegrityClient.integrityClientPaths,
    OccupancyClient = ecalOccupancyClient.occupancyClientPaths,
    PresampleClient = ecalPresampleClient.presampleClientPaths,
    TrigPrimClient = ecalTrigPrimClient.trigPrimClientPaths,
    RawDataClient = ecalRawDataClient.rawDataClientPaths,
    TimingClient = ecalTimingClient.timingClientPaths,
    SelectiveReadoutClient = ecalSelectiveReadoutClient.selectiveReadoutClientPaths,
    SummaryClient = ecalSummaryClient.summaryClientPaths
)

ecalMonitorClientParams = dict(
    IntegrityClient = ecalIntegrityClient.integrityClient,
    OccupancyClient = ecalOccupancyClient.occupancyClient,
    PresampleClient = ecalPresampleClient.presampleClient,
    TrigPrimClient = ecalTrigPrimClient.trigPrimClient,
    RawDataClient = ecalRawDataClient.rawDataClient,
    TimingClient = ecalTimingClient.timingClient,
    SelectiveReadoutClient = ecalSelectiveReadoutClient.selectiveReadoutClient,
    SummaryClient = ecalSummaryClient.summaryClient,
    Common = ecalCommonParams
)

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
    workerParameters = dqmpset(ecalMonitorClientParams),
    # ME paths for each task (included from inidividual cfis)
    mePaths = dqmpaths("Ecal", ecalMonitorClientPaths),
    verbosity = cms.untracked.int32(0)
)
