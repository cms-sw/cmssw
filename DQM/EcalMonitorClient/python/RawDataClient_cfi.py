import FWCore.ParameterSet.Config as cms

from DQM.EcalMonitorTasks.OccupancyTask_cfi import ecalOccupancyTask
from DQM.EcalMonitorTasks.RawDataTask_cfi import ecalRawDataTask

synchErrThresholdFactor = 1.

ecalRawDataClient = cms.untracked.PSet(
    params = cms.untracked.PSet(
        synchErrThresholdFactor = cms.untracked.double(synchErrThresholdFactor)
    ),
    sources = cms.untracked.PSet(
        Entries = ecalOccupancyTask.MEs.DCC,
        L1ADCC = ecalRawDataTask.MEs.L1ADCC,
        FEStatus = ecalRawDataTask.MEs.FEStatus
    ),
    MEs = cms.untracked.PSet(
        QualitySummary = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sSFT%(suffix)s front-end status summary'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('SuperCrystal'),
            description = cms.untracked.string('Summary of the raw data (DCC and front-end) quality. A channel is red if it has nonzero events with FE status that is different from any of ENABLED, DISABLED, SUPPRESSED, FIFOFULL, FIFOFULL_L1ADESYNC, and FORCEDZS. A FED can also go red if its number of L1A desynchronization errors is greater than ' + str(synchErrThresholdFactor) + ' * log10(total entries).')
        ),
        ErrorsSummary = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sSFT front-end status errors summary'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('DCC'),
            description = cms.untracked.string('Counter of data towers flagged as bad in the quality summary')
        )
    )
)

