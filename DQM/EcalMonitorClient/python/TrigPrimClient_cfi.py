import FWCore.ParameterSet.Config as cms

from DQM.EcalMonitorTasks.TrigPrimTask_cfi import ecalTrigPrimTask
from DQM.EcalMonitorTasks.OccupancyTask_cfi import ecalOccupancyTask

minEntries = 3
errorFractionThreshold = 0.2
TTF4MaskingAlarmThreshold = 0.1

ecalTrigPrimClient = cms.untracked.PSet(
    params = cms.untracked.PSet(
        minEntries = cms.untracked.int32(minEntries),
        errorFractionThreshold = cms.untracked.double(errorFractionThreshold),
        TTF4MaskingAlarmThreshold = cms.untracked.double(TTF4MaskingAlarmThreshold),
        sourceFromEmul = cms.untracked.bool(True)
    ),
    sources = cms.untracked.PSet(
        EtEmulError = ecalTrigPrimTask.MEs.EtEmulError,
        MatchedIndex = ecalTrigPrimTask.MEs.MatchedIndex,
        TTFlags4 = ecalTrigPrimTask.MEs.TTFlags4,
        TTFlags4ByLumi = ecalTrigPrimTask.MEs.TTFlags4ByLumi,
        TTMaskMapAll = ecalTrigPrimTask.MEs.TTMaskMapAll,
        TTFMismatch = ecalTrigPrimTask.MEs.TTFMismatch,
        LHCStatusByLumi = ecalTrigPrimTask.MEs.LHCStatusByLumi,
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
            description = cms.untracked.string('Summary of emulator matching quality. A tower is red if the number of events with Et emulation error is greater than ' + str(errorFractionThreshold) + ' of total events. Towers with entries less than ' + str(minEntries) + ' are not considered. Also, an entire SuperModule can show red if its (data) Trigger Primitive digi occupancy is less than 5sigma of the overall SuperModule mean, or if more than 80% of its Trigger Towers are showing any number of TT Flag-Readout Mismatch errors. Finally, if the fraction of towers in a SuperModule that are permanently masked or have TTF4 flag set is greater than ' + str(TTF4MaskingAlarmThreshold) + ', then the entire supermodule shows red.')
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
        ),
        TTF4vMaskByLumi = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sTriggerTowerTask/%(prefix)sTTT TTF4 vs Masking Status%(suffix)s by lumi'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('TriggerTower'),
            description = cms.untracked.string('Summarizes whether a TT was masked in this lumisection in the TPGRecords, or had an instance of TT Flag=4.<br/>GRAY: Masked, no TTF4,<br/>BLACK: Masked, with TTF4,<br/>BLUE: Not Masked, with TTF4.')
        ),
        TrendTTF4Flags = cms.untracked.PSet(
            path = cms.untracked.string('Ecal/Trends/TrigPrimClient %(prefix)s number of TTs with TTF4 set'),
            kind = cms.untracked.string('TProfile'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('Trend'),
            description = cms.untracked.string('Trend of the total number of TTs in this partition with TTF4 flag set.')
        )
    )
)
