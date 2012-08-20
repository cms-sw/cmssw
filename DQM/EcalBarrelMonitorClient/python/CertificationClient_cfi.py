certificationClient = dict(
    sources = dict(
        DAQ = ['TowerStatusTask', 'DAQSummaryMap'],
        DCS = ['TowerStatusTask', 'DCSSummaryMap'],
        Report = ['SummaryClient', 'ReportSummaryMap']
    )
)

certificationClientPaths = dict(
    CertificationMap = "EventInfo/CertificationSummaryMap",
    CertificationContents = "EventInfo/CertificationContents/",
    Certification = "EventInfo/CertificationSummary",
)
