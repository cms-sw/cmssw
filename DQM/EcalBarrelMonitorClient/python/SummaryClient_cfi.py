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
        QualitySummary = dict(path = "%(subdet)s/%(prefix)sSummaryClient/%(prefix)s global summary%(suffix)s", otype = 'Ecal3P', btype = 'Crystal', kind = 'TH2F'),
        ReportSummaryMap = dict(path = "Ecal/EventInfo/reportSummaryMap", otype = 'Ecal', btype = 'DCC', kind = 'TH2F'),
        ReportSummaryContents = dict(path = "Ecal/EventInfo/reportSummaryContents/Ecal_%(sm)s", otype = 'SM', btype = 'Report', kind = 'REAL'),
        ReportSummary = dict(path = "Ecal/EventInfo/reportSummary", otype = 'Ecal', btype = 'Report', kind = 'REAL'),
        NBadFEDs = dict(path = "Ecal/Errors/Number of Bad Ecal FEDs", otype = 'None', btype = 'User', kind = 'TH1F', xaxis = {'nbins': 1, 'low': 0., 'high': 1.})
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
