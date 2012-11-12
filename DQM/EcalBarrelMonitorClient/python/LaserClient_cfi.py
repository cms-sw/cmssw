import FWCore.ParameterSet.Config as cms

from DQM.EcalBarrelMonitorTasks.LaserTask_cfi import ecalLaserTask

forwardFactor = 0.5
minChannelEntries = 3
expectedAmplitude = [1700.0, 1300.0, 1700.0, 1700.0]
toleranceAmplitude = 0.1
toleranceAmpRMSRatio = 0.3
expectedPNAmplitude = [800.0, 800.0, 800.0, 800.0]
tolerancePNAmp = 0.1
tolerancePNRMSRatio = 1.
expectedTiming = [4.2, 4.2, 4.2, 4.2]
toleranceTiming = 0.5
toleranceTimRMS = 0.4

ecalLaserClient = cms.untracked.PSet(
    forwardFactor = cms.untracked.double(forwardFactor),
    minChannelEntries = cms.untracked.int32(minChannelEntries),
    expectedAmplitude = cms.untracked.vdouble(expectedAmplitude),
    toleranceAmplitude = cms.untracked.double(toleranceAmplitude),
    toleranceAmpRMSRatio = cms.untracked.double(toleranceAmpRMSRatio),
    expectedPNAmplitude = cms.untracked.vdouble(expectedPNAmplitude),
    tolerancePNAmp = cms.untracked.double(tolerancePNAmp),
    tolerancePNRMSRatio = cms.untracked.double(tolerancePNRMSRatio),
    expectedTiming = cms.untracked.vdouble(expectedTiming),
    toleranceTiming = cms.untracked.double(toleranceTiming),    
    toleranceTimRMS = cms.untracked.double(toleranceTimRMS),
    sources = cms.untracked.PSet(
        Timing = ecalLaserTask.MEs.Timing,
        PNAmplitude = ecalLaserTask.MEs.PNAmplitude,
        Amplitude = ecalLaserTask.MEs.Amplitude
    ),
    MEs = cms.untracked.PSet(
        TimingRMS = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            multi = cms.untracked.int32(4),
            otype = cms.untracked.string('SM'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(0.5),
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('rms (clock)')
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sLaserClient/%(prefix)sLT laser timing rms L%(wl)s %(sm)s'),
            description = cms.untracked.string('Distribution of the timing RMS in each crystal channel. X scale is in LHC clocks. Channels with less than ' + str(minChannelEntries) + ' are not considered.')
        ),
        TimingMean = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            multi = cms.untracked.int32(4),
            otype = cms.untracked.string('SM'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(5.5),
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(3.5),
                title = cms.untracked.string('time (clock)')
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sLaserClient/%(prefix)sLT laser timing mean L%(wl)s %(sm)s'),
            description = cms.untracked.string('Distribution of the timing in each crystal channel. X scale is in LHC clocks. Channels with less than ' + str(minChannelEntries) + ' are not considered.')
        ),
        PNQualitySummary = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sLT PN laser quality summary L%(wl)s'),
            otype = cms.untracked.string('MEM2P'),
            multi = cms.untracked.int32(4),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('Crystal'),
            description = cms.untracked.string('Summary of the laser data quality in the PN diodes. A channel is red if mean / expected < ' + str(tolerancePNAmp) + ' or RMS / expected > ' + str(tolerancePNRMSRatio) + '. Expected amplitudes are ' + ('%.1f, %.1f, %.1f, %.1f' % tuple(expectedPNAmplitude)) + ' for laser 1, 2, 3, and 4 respectively. Channels with less than ' + str(minChannelEntries) + ' are not considered.'),
        ),
        TimingRMSMap = cms.untracked.PSet(
            multi = cms.untracked.int32(4),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('Ecal2P'),
            zaxis = cms.untracked.PSet(
                title = cms.untracked.string('rms (clock)')
            ),
            btype = cms.untracked.string('Crystal'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sLaserClient/%(prefix)sLT laser timing rms map L%(wl)s'),
            description = cms.untracked.string('2D distribution of the laser timing RMS. Z scale is in LHC clocks. Channels with less than ' + str(minChannelEntries) + ' are not considered.')
        ),
        AmplitudeMean = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            multi = cms.untracked.int32(4),
            otype = cms.untracked.string('SM'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(4096.0),
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0.0)
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sLaserClient/%(prefix)sLT amplitude L%(wl)s %(sm)s'),
            description = cms.untracked.string('Distribution of the mean amplitude seen in each crystal. Channels with less than ' + str(minChannelEntries) + ' are not considered.')
        ),
        QualitySummary = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sLT%(suffix)s laser quality summary L%(wl)s'),
            otype = cms.untracked.string('Ecal3P'),
            multi = cms.untracked.int32(4),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('SuperCrystal'),
            description = cms.untracked.string('Summary of the laser data quality. A channel is red either if mean / expected < ' + str(toleranceAmplitude) + ', or if RMS / expected > ' + str(toleranceAmpRMSRatio) + ', or if mean timing is off from expected by ' + str(toleranceTiming) + '. Expected amplitudes and timings are ' + ('%.1f, %.1f, %.1f, %.1f' % tuple(expectedAmplitude)) + ' and ' + ('%.1f, %.1f, %.1f, %.1f' % tuple(expectedTiming)) + ' for lasers 1, 2, 3, and 4 respectively. Channels with less than ' + str(minChannelEntries) + ' are not considered.'),
        ),
        Quality = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sLaserClient/%(prefix)sLT laser quality L%(wl)s %(sm)s'),
            otype = cms.untracked.string('SM'),
            multi = cms.untracked.int32(4),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('Crystal'),
            description = cms.untracked.string('Summary of the laser data quality. A channel is red either if mean / expected < ' + str(toleranceAmplitude) + ', or if RMS / expected > ' + str(toleranceAmpRMSRatio) + ', or if mean timing is off from expected by ' + str(toleranceTiming) + '. Expected amplitudes and timings are ' + ('%.1f, %.1f, %.1f, %.1f' % tuple(expectedAmplitude)) + ' and ' + ('%.1f, %.1f, %.1f, %.1f' % tuple(expectedTiming)) + ' for lasers 1, 2, 3, and 4 respectively. Channels with less than ' + str(minChannelEntries) + ' are not considered.'),
        ),
        AmplitudeRMS = cms.untracked.PSet(
            multi = cms.untracked.int32(4),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('Ecal2P'),
            zaxis = cms.untracked.PSet(
                title = cms.untracked.string('rms (ADC counts)')
            ),
            btype = cms.untracked.string('Crystal'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sLaserClient/%(prefix)sLT amplitude rms L%(wl)s'),
            description = cms.untracked.string('2D distribution of the amplitude RMS. Channels with less than ' + str(minChannelEntries) + ' are not considered.')
        )
    )
)
