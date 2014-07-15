import FWCore.ParameterSet.Config as cms

from DQM.EcalCommon.CommonParams_cfi import ecalCommonParams

from DQM.EcalMonitorClient.IntegrityClient_cfi import ecalIntegrityClient
from DQM.EcalMonitorClient.OccupancyClient_cfi import ecalOccupancyClient
from DQM.EcalMonitorClient.PresampleClient_cfi import ecalPresampleClient
from DQM.EcalMonitorClient.RawDataClient_cfi import ecalRawDataClient
from DQM.EcalMonitorClient.SelectiveReadoutClient_cfi import ecalSelectiveReadoutClient
from DQM.EcalMonitorClient.TimingClient_cfi import ecalTimingClient
from DQM.EcalMonitorClient.TrigPrimClient_cfi import ecalTrigPrimClient
from DQM.EcalMonitorClient.SummaryClient_cfi import ecalSummaryClient

ecalMonitorClient = cms.EDAnalyzer("EcalDQMonitorClient",
    moduleName = cms.untracked.string("Ecal Monitor Client"),
    # workers to be turned on
    workers = cms.untracked.vstring(
        "IntegrityClient",
        "OccupancyClient",
        "PresampleClient",
        "RawDataClient",
        "TrigPrimClient",
        "SummaryClient"
    ),
    # task parameters (included from indivitual cfis)
    workerParameters = cms.untracked.PSet(
        IntegrityClient = ecalIntegrityClient,
        OccupancyClient = ecalOccupancyClient,
        PresampleClient = ecalPresampleClient,
        RawDataClient = ecalRawDataClient,
        SelectiveReadoutClient = ecalSelectiveReadoutClient,
        TimingClient = ecalTimingClient,
        TrigPrimClient = ecalTrigPrimClient,
        SummaryClient = ecalSummaryClient
    ),
    commonParameters = ecalCommonParams,
    verbosity = cms.untracked.int32(0)
)

