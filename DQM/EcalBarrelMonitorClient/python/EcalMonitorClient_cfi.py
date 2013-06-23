import FWCore.ParameterSet.Config as cms

from DQM.EcalCommon.dqmpset import *
from DQM.EcalCommon.CommonParams_cfi import *

from DQM.EcalBarrelMonitorTasks.EcalMonitorTask_cfi import ecalMonitorTaskPaths

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

ecalMonitorClientSources = dict(ecalMonitorClientPaths)
ecalMonitorClientSources.update(ecalMonitorTaskPaths)

ecalMonitorClientParams = dict(
    IntegrityClient = ecalIntegrityClient.integrityClient,
    OccupancyClient = ecalOccupancyClient.occupancyClient,
    PresampleClient = ecalPresampleClient.presampleClient,
    TrigPrimClient = ecalTrigPrimClient.trigPrimClient,
    RawDataClient = ecalRawDataClient.rawDataClient,
    TimingClient = ecalTimingClient.timingClient,
    SelectiveReadoutClient = ecalSelectiveReadoutClient.selectiveReadoutClient,
    SummaryClient = ecalSummaryClient.summaryClient,
    Common = ecalCommonParams,
    sources = dqmpaths("Ecal", ecalMonitorClientSources)
)

ecalMonitorClient = cms.EDAnalyzer("EcalDQMonitorClient",
    moduleName = cms.untracked.string("Ecal Monitor Client"),
    # clients to be turned on
    clients = cms.untracked.vstring(
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
    clientParameters = dqmpset(ecalMonitorClientParams),
    # ME paths for each task (included from inidividual cfis)
    mePaths = dqmpaths("Ecal", ecalMonitorClientPaths),
    runAtEndLumi = cms.untracked.bool(False),
    verbosity = cms.untracked.int32(0)
)
