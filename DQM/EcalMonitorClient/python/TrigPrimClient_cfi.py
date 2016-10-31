import FWCore.ParameterSet.Config as cms

from DQM.EcalMonitorTasks.TrigPrimTask_cfi import ecalTrigPrimTask
from DQM.EcalMonitorTasks.OccupancyTask_cfi import ecalOccupancyTask

minEntries = 3
errorFractionThreshold = 0.1

ecalTrigPrimClient = cms.untracked.PSet(
    params = cms.untracked.PSet(
        minEntries = cms.untracked.int32(minEntries),
        errorFractionThreshold = cms.untracked.double(errorFractionThreshold)
    ),
    sources = cms.untracked.PSet(
        EtEmulError = ecalTrigPrimTask.MEs.EtEmulError,
        MatchedIndex = ecalTrigPrimTask.MEs.MatchedIndex,
        TTFlags4 = ecalTrigPrimTask.MEs.TTFlags4,
        TTMaskMapAll = ecalTrigPrimTask.MEs.TTMaskMapAll,
        TTFMismatch = ecalTrigPrimTask.MEs.TTFMismatch,
        TPDigiThrAll = ecalOccupancyTask.MEs.TPDigiThrAll
    ),
    MEs = cms.untracked.PSet(
        NonSingleSummary = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sTTT%(suffix)s Trigger Primitives Non Single Timing summary'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('TriggerTower'),
            zaxis = cms.untracked.PSet(
                title = cms.untracked.string('rate')
            ),
            description = cms.untracked.string('Fraction of events whose emulator TP timing did not agree with the majority. Towers with entries less than ' + str(minEntries) + ' are not considered.')
        ),
        EmulQualitySummary = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sTTT%(suffix)s emulator error quality summary'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('TriggerTower'),
            description = cms.untracked.string('Summary of emulator matching quality. A tower is red if the number of events with Et emulation error is greater than ' + str(errorFractionThreshold) + ' of total events. Towers with entries less than ' + str(minEntries) + ' are not considered.')
        ),
        TimingSummary = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sTTT%(suffix)s Trigger Primitives Timing summary'),
            kind = cms.untracked.string('TH2F'),
            zaxis = cms.untracked.PSet(
                title = cms.untracked.string('TP data matching emulator')
            ),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('TriggerTower'),
            description = cms.untracked.string('Emulator TP timing where the largest number of events had Et matches. Towers with entries less than ' + str(minEntries) + ' are not considered.')
        ),
        TTF4vMask = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sTriggerTowerTask/%(prefix)sTTT TTF4 vs Masking Status%(suffix)s'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('TriggerTower'),
            description = cms.untracked.string('Summarizes whether a TT was masked in the TPGRecords, or had an instance of TT Flag=4.<br/>GRAY: Masked, no TTF4,<br/>BLACK: Masked, with TTF4,<br/>BLUE: Not Masked, with TTF4.')
        )
    )
)
