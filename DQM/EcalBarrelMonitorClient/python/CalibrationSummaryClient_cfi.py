from DQM.EcalBarrelMonitorClient.PNIntegrityClient_cfi import pnIntegrityClient
from DQM.EcalBarrelMonitorClient.LaserClient_cfi import laserClient
from DQM.EcalBarrelMonitorClient.LedClient_cfi import ledClient
from DQM.EcalBarrelMonitorClient.TestPulseClient_cfi import testPulseClient
from DQM.EcalBarrelMonitorClient.PedestalClient_cfi import pedestalClient

calibrationSummaryClient = dict(
#    laserWavelengths = [1, 2, 3, 4],
    laserWavelengths = [3],
#    ledWavelengths = [1, 2],
#    testPulseMGPAGains = [1, 6, 12],
    testPulseMGPAGains = [12],
#    testPulseMGPAGainsPN = [1, 16],
    testPulseMGPAGainsPN = [16],
#    pedestalMGPAGains = [1, 6, 12],
#    pedestalMGPAGainsPN = [1, 16],
    MEs = dict(
        QualitySummary = dict(path = 'Summary/CalibSummaryClient global quality', otype = 'Ecal2P', btype = 'SuperCrystal', kind = 'TH2F'),
        PNQualitySummary = dict(path = 'Summary/CalibSummaryClient PN global quality', otype = 'MEM', btype = 'Crystal', kind = 'TH2F'),
        ReportSummaryMap = dict(path = 'EventInfo/reporSummaryMap', otype = 'Ecal', btype = 'DCC', kind = 'TH2F'),
        ReportSummaryContents = dict(path = "EventInfo/reportSummaryContents/", otype = 'SM', btype = 'Report', kind = 'REAL'),
        ReportSummary = dict(path = "EventInfo/reportSummary", otype = 'Ecal', btype = 'Report', kind = 'REAL')
    ),
    sources = dict(
        PNIntegrity = pnIntegrityClient['MEs']['QualitySummary'],
        Laser = laserClient['MEs']['QualitySummary'],
        LaserPN = laserClient['MEs']['PNQualitySummary'],
        Led = ledClient['MEs']['QualitySummary'],
        LedPN = ledClient['MEs']['PNQualitySummary'],
        TestPulse = testPulseClient['MEs']['QualitySummary'],
        TestPulsePN = testPulseClient['MEs']['PNQualitySummary'],
        Pedestal = pedestalClient['MEs']['QualitySummary'],
        PedestalPN = pedestalClient['MEs']['PNQualitySummary']
    )
)
    
