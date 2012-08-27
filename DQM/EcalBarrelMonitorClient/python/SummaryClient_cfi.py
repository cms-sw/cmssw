from DQM.EcalBarrelMonitorClient.IntegrityClient_cfi import integrityClient
from DQM.EcalBarrelMonitorClient.PresampleClient_cfi import presampleClient
from DQM.EcalBarrelMonitorClient.TimingClient_cfi import timingClient
from DQM.EcalBarrelMonitorClient.RawDataClient_cfi import rawDataClient
from DQM.EcalBarrelMonitorTasks.OccupancyTask_cfi import occupancyTask

summaryClient = dict(
    MEs = dict(
        QualitySummary = dict(path = "Summary/SummaryClient global quality", otype = 'Ecal2P', btype = 'Crystal', kind = 'TH2F'),
        ReportSummaryMap = dict(path = "EventInfo/reportSummaryMap", otype = 'Ecal', btype = 'DCC', kind = 'TH2F'),
        ReportSummaryContents = dict(path = "EventInfo/reportSummaryContents/", otype = 'SM', btype = 'Report', kind = 'REAL'),
        ReportSummary = dict(path = "EventInfo/reportSummary", otype = 'Ecal', btype = 'Report', kind = 'REAL')
    ),
    sources = dict(
        Integrity = integrityClient['MEs']['Quality'],
        Presample = presampleClient['MEs']['Quality'],
        Timing = timingClient['MEs']['Quality'],
        RawData = rawDataClient['MEs']['QualitySummary']
    )
)
