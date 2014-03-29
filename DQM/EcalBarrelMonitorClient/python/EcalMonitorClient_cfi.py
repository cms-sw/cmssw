import FWCore.ParameterSet.Config as cms

from DQM.EcalCommon.CommonParams_cfi import ecalCommonParams

from DQM.EcalBarrelMonitorClient.IntegrityClient_cfi import ecalIntegrityClient
from DQM.EcalBarrelMonitorClient.OccupancyClient_cfi import ecalOccupancyClient
from DQM.EcalBarrelMonitorClient.PresampleClient_cfi import ecalPresampleClient
from DQM.EcalBarrelMonitorClient.RawDataClient_cfi import ecalRawDataClient
from DQM.EcalBarrelMonitorClient.SelectiveReadoutClient_cfi import ecalSelectiveReadoutClient
from DQM.EcalBarrelMonitorClient.TimingClient_cfi import ecalTimingClient
from DQM.EcalBarrelMonitorClient.TrigPrimClient_cfi import ecalTrigPrimClient
from DQM.EcalBarrelMonitorClient.SummaryClient_cfi import ecalSummaryClient

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

