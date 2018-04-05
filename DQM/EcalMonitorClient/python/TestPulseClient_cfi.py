import FWCore.ParameterSet.Config as cms

from DQM.EcalCommon.CommonParams_cfi import *

from DQM.EcalMonitorTasks.TestPulseTask_cfi import ecalTestPulseTask

minChannelEntries = 3
amplitudeThreshold = [1200., 600., 100.]
toleranceRMS = [160., 80., 10.]
PNAmplitudeThreshold = [12.5, 200.]
tolerancePNRMS = [20., 20.]

ecalTestPulseClient = cms.untracked.PSet(
    params = cms.untracked.PSet(
        minChannelEntries = cms.untracked.int32(minChannelEntries),
        amplitudeThreshold = cms.untracked.vdouble(amplitudeThreshold),
        toleranceRMS = cms.untracked.vdouble(toleranceRMS),
        PNAmplitudeThreshold = cms.untracked.vdouble(PNAmplitudeThreshold),
        tolerancePNRMS = cms.untracked.vdouble(tolerancePNRMS),
        MGPAGains = ecaldqmMGPAGains,
        MGPAGainsPN = ecaldqmMGPAGainsPN
    ),
    sources = cms.untracked.PSet(
        Amplitude = ecalTestPulseTask.MEs.Amplitude,
        PNAmplitude = ecalTestPulseTask.MEs.PNAmplitude
    ),
    MEs = cms.untracked.PSet(
        PNQualitySummary = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sTPT PN test pulse quality G%(pngain)s summary'),
            otype = cms.untracked.string('MEM2P'),
            multi = cms.untracked.PSet(
                pngain = ecaldqmMGPAGainsPN
            ),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('Crystal'),
            description = cms.untracked.string('Summary of test pulse data quality for PN diodes. A channel is red if mean amplitude is lower than the threshold, or RMS is greater than threshold. The mean and RMS thresholds are ' + ('%.1f, %.1f' % tuple(PNAmplitudeThreshold)) + ' and ' + ('%.1f, %.1f' % tuple(tolerancePNRMS)) + ' for gains 1 and 16 respectively. Channels with entries less than ' + str(minChannelEntries) + ' are not considered.')
        ),
        QualitySummary = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sTPT%(suffix)s test pulse quality G%(gain)s summary'),
            otype = cms.untracked.string('Ecal3P'),
            multi = cms.untracked.PSet(
                gain = ecaldqmMGPAGains
            ),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('SuperCrystal'),
            description = cms.untracked.string('Summary of test pulse data quality for crystals. A channel is red if mean amplitude is lower than the threshold, or RMS is greater than threshold. The mean and RMS thresholds are ' + ('%.1f, %.1f, %.1f' % tuple(amplitudeThreshold)) + ' and ' + ('%.1f, %.1f, %.1f' % tuple(toleranceRMS)) + ' for gains 1, 6, and 12 respectively. Channels with entries less than ' + str(minChannelEntries) + ' are not considered.')
        ),
        Quality = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sTestPulseClient/%(prefix)sTPT test pulse quality G%(gain)s %(sm)s'),
            otype = cms.untracked.string('SM'),
            multi = cms.untracked.PSet(
                gain = ecaldqmMGPAGains
            ),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('Crystal'),
            description = cms.untracked.string('Summary of test pulse data quality for crystals. A channel is red if mean amplitude is lower than the threshold, or RMS is greater than threshold. The mean and RMS thresholds are ' + ('%.1f, %.1f, %.1f' % tuple(amplitudeThreshold)) + ' and ' + ('%.1f, %.1f, %.1f' % tuple(toleranceRMS)) + ' for gains 1, 6, and 12 respectively. Channels with entries less than ' + str(minChannelEntries) + ' are not considered.')
        ),
        AmplitudeRMS = cms.untracked.PSet(
            multi = cms.untracked.PSet(
                gain = ecaldqmMGPAGains
            ),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('SM'),
            zaxis = cms.untracked.PSet(
                title = cms.untracked.string('rms (ADC counts)')
            ),
            btype = cms.untracked.string('Crystal'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sTestPulseClient/%(prefix)sTPT test pulse rms G%(gain)s %(sm)s'),
            description = cms.untracked.string('2D distribution of the amplitude RMS. Channels with entries less than ' + str(minChannelEntries) + ' are not considered.')
        )
    )
)
