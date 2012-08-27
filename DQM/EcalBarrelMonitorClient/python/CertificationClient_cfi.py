from DQM.EcalBarrelMonitorTasks.TowerStatusTask_cfi import towerStatusTask
from DQM.EcalBarrelMonitorClient.SummaryClient_cfi import summaryClient

certificationClient = dict(
    MEs = dict(
        CertificationMap = dict(path = "EventInfo/CertificationSummaryMap", otype = 'Ecal', btype = 'SuperCrystal', kind = 'TH2F'),
        CertificationContents = dict(path = "EventInfo/CertificationContents/", otype  = 'SM', btype = 'Report', kind = 'REAL'),
        Certification = dict(path = "EventInfo/CertificationSummary", otype = 'Ecal', btype = 'Report', kind = 'REAL')
    ),
    sources = dict(
        DAQ = towerStatusTask['MEs']['DAQSummaryMap'],
        DCS = towerStatusTask['MEs']['DCSSummaryMap'],
        Report = summaryClient['MEs']['ReportSummaryMap']
    )
)
