from DQM.EcalBarrelMonitorTasks.TowerStatusTask_cfi import ecalTowerStatusTask
from DQM.EcalBarrelMonitorClient.SummaryClient_cfi import ecalSummaryClient

ecalCertificationClient = dict(
    MEs = dict(
        CertificationMap = dict(path = "Ecal/EventInfo/CertificationSummaryMap", otype = 'Ecal', btype = 'DCC', kind = 'TH2F'),
        CertificationContents = dict(path = "Ecal/EventInfo/CertificationContents/Ecal_%(sm)s", otype  = 'SM', btype = 'Report', kind = 'REAL'),
        Certification = dict(path = "Ecal/EventInfo/CertificationSummary", otype = 'Ecal', btype = 'Report', kind = 'REAL')
    ),
    sources = dict(
        DAQ = ecalTowerStatusTask['MEs']['DAQContents'],
        DCS = ecalTowerStatusTask['MEs']['DCSContents'],
        DQM = ecalSummaryClient['MEs']['ReportSummaryContents']
    )
)
