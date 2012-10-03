import FWCore.ParameterSet.Config as cms

from DQM.EcalBarrelMonitorClient.IntegrityClient_cfi import ecalIntegrityClient
from DQM.EcalBarrelMonitorClient.PresampleClient_cfi import ecalPresampleClient
from DQM.EcalBarrelMonitorClient.TimingClient_cfi import ecalTimingClient
from DQM.EcalBarrelMonitorClient.RawDataClient_cfi import ecalRawDataClient
from DQM.EcalBarrelMonitorClient.TrigPrimClient_cfi import ecalTrigPrimClient
from DQM.EcalBarrelMonitorClient.OccupancyClient_cfi import ecalOccupancyClient
from DQM.EcalBarrelMonitorTasks.IntegrityTask_cfi import ecalIntegrityTask
from DQM.EcalBarrelMonitorTasks.RawDataTask_cfi import ecalRawDataTask

activeSources = ['Integrity', 'RawData', 'Presample', 'Timing']
towerBadFraction = 0.8
fedBadFraction = 0.5

ecalSummaryClient = cms.untracked.PSet(
    activeSources = cms.untracked.vstring(activeSources),
    towerBadFraction = cms.untracked.double(towerBadFraction),
    fedBadFraction = cms.untracked.double(fedBadFraction),
    sources = cms.untracked.PSet(
        Integrity = ecalIntegrityClient.MEs.QualitySummary,
        IntegrityByLumi = ecalIntegrityTask.MEs.ByLumi,
        Presample = ecalPresampleClient.MEs.QualitySummary,
        Timing = ecalTimingClient.MEs.QualitySummary,
        RawData = ecalRawDataClient.MEs.QualitySummary,
        DesyncByLumi = ecalRawDataTask.MEs.DesyncByLumi,
        FEByLumi = ecalRawDataTask.MEs.FEByLumi,
        TriggerPrimitives = ecalTrigPrimClient.MEs.EmulQualitySummary,
        HotCell = ecalOccupancyClient.MEs.QualitySummary
    ),
    MEs = cms.untracked.PSet(
        ReportSummaryMap = cms.untracked.PSet(
            path = cms.untracked.string('Ecal/EventInfo/reportSummaryMap'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('Ecal'),
            btype = cms.untracked.string('DCC'),
            description = cms.untracked.string('')
        ),
        ReportSummaryContents = cms.untracked.PSet(
            path = cms.untracked.string('Ecal/EventInfo/reportSummaryContents/Ecal_%(sm)s'),
            kind = cms.untracked.string('REAL'),
            otype = cms.untracked.string('SM'),
            btype = cms.untracked.string('Report'),
            description = cms.untracked.string('')
        ),
        NBadFEDs = cms.untracked.PSet(
            path = cms.untracked.string('Ecal/Errors/Number of Bad Ecal FEDs'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('None'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(1.0),
                nbins = cms.untracked.int32(1),
                low = cms.untracked.double(0.0)
            ),
            btype = cms.untracked.string('User'),
            description = cms.untracked.string('Number of FEDs with more than ' + str(fedBadFraction * 100) + '% of channels in bad status. Updated every lumi section.')
        ),
        QualitySummary = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)s global summary%(suffix)s'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('Crystal'),
            description = cms.untracked.string('Summary of the data quality. A channel is red if it is red in any one of RawData, Integrity, Timing, TriggerPrimitives, and HotCells task. A cluster of bad towers in this plot will cause the ReportSummary for the FED to go to 0 in online DQM.')
        ),
        ReportSummary = cms.untracked.PSet(
            path = cms.untracked.string('Ecal/EventInfo/reportSummary'),
            kind = cms.untracked.string('REAL'),
            otype = cms.untracked.string('Ecal'),
            btype = cms.untracked.string('Report'),
            description = cms.untracked.string('')
        )
    )
)
