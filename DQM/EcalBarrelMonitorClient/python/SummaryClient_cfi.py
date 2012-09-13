from DQM.EcalBarrelMonitorClient.IntegrityClient_cfi import ecalIntegrityClient
from DQM.EcalBarrelMonitorClient.PresampleClient_cfi import ecalPresampleClient
from DQM.EcalBarrelMonitorClient.TimingClient_cfi import ecalTimingClient
from DQM.EcalBarrelMonitorClient.RawDataClient_cfi import ecalRawDataClient
from DQM.EcalBarrelMonitorClient.TrigPrimClient_cfi import ecalTrigPrimClient
from DQM.EcalBarrelMonitorClient.OccupancyClient_cfi import ecalOccupancyClient
from DQM.EcalBarrelMonitorTasks.IntegrityTask_cfi import ecalIntegrityTask
from DQM.EcalBarrelMonitorTasks.RawDataTask_cfi import ecalRawDataTask

ecalSummaryClient = dict(
    activeSources = ['Integrity', 'RawData', 'Presample', 'Timing'],
    MEs = dict(
        QualitySummary = dict(path = "Summary/SummaryClient global quality", otype = 'Ecal2P', btype = 'Crystal', kind = 'TH2F'),
        ReportSummaryMap = dict(path = "EventInfo/reportSummaryMap", otype = 'Ecal', btype = 'DCC', kind = 'TH2F'),
        ReportSummaryContents = dict(path = "EventInfo/reportSummaryContents/", otype = 'SM', btype = 'Report', kind = 'REAL'),
        ReportSummary = dict(path = "EventInfo/reportSummary", otype = 'Ecal', btype = 'Report', kind = 'REAL')
    ),
    sources = dict(
        Integrity = ecalIntegrityClient['MEs']['QualitySummary'],
        IntegrityByLumi = ecalIntegrityTask['MEs']['ByLumi'],
        Presample = ecalPresampleClient['MEs']['QualitySummary'],
        Timing = ecalTimingClient['MEs']['QualitySummary'],
        RawData = ecalRawDataClient['MEs']['QualitySummary'],
        DesyncByLumi = ecalRawDataTask['MEs']['DesyncByLumi'],
        FEByLumi = ecalRawDataTask['MEs']['FEByLumi'],
        TriggerPrimitives = ecalTrigPrimClient['MEs']['EmulQualitySummary'],
        HotCell = ecalOccupancyClient['MEs']['QualitySummary']
    )
)
