summaryClient = dict(
    sources = dict(
        Integrity = ['IntegrityClient', 'Quality'],
        Presample = ['PresampleClient', 'Quality'],
        Timing = ['TimingClient', 'Quality'],
        RawData = ['RawDataClient', 'QualitySummary'],
        DigiOccupancy = ['OccupancyTask', 'Digi']
    )
)

summaryClientPaths = dict(
    QualitySummary = "Summary/SummaryClient global quality",
    ReportSummaryMap = "EventInfo/reportSummaryMap",
    ReportSummaryContents = "EventInfo/reportSummaryContents/",
    ReportSummary = "EventInfo/reportSummary"
)
