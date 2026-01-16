import FWCore.ParameterSet.Config as cms

from DQM.EcalMonitorClient.EcalMonitorClient_cfi import *

ecalMonitorClientPhase2 = ecalMonitorClient.clone(
    workers = cms.untracked.vstring(
        "OccupancyClient",
        "TimingClient",
        "SummaryClient"
    ),
    # task parameters (included from indivitual cfis)
    workerParameters = cms.untracked.PSet(
        OccupancyClient = ecalOccupancyClient,
        TimingClient = ecalTimingClient,
        SummaryClient = ecalSummaryClient
    ),
    commonParameters = ecalCommonParams,
    verbosity = cms.untracked.int32(0)
)

