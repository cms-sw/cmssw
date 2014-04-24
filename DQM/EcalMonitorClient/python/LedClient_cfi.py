import FWCore.ParameterSet.Config as cms

from DQM.EcalCommon.CommonParams_cfi import *

from DQM.EcalMonitorTasks.LedTask_cfi import ecalLedTask

forwardFactor = 0.5
minChannelEntries = 3
expectedAmplitude = [200., 10.]
toleranceAmplitude = 0.1
toleranceAmpRMSRatio = 0.5
expectedTiming = [4.4, 4.5]
toleranceTiming = 1.
toleranceTimRMS = 25.
expectedPNAmplitude = [800., 800.]
tolerancePNAmp = 0.1
tolerancePNRMSRatio = 1.

ecalLedClient = cms.untracked.PSet(
    params = cms.untracked.PSet(
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
        toleranceTimRMS = cms.untracked.double(toleranceTimRMS)
    ),
    sources = cms.untracked.PSet(
        Timing = ecalLedTask.MEs.Timing,
        PNAmplitude = ecalLedTask.MEs.PNAmplitude,
        Amplitude = ecalLedTask.MEs.Amplitude
    ),
    MEs = cms.untracked.PSet(
        TimingMean = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            multi = cms.untracked.PSet(
                wl = ecaldqmLedWavelengths
            ),
            otype = cms.untracked.string('EESM'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(5.5),
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(3.5)
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('EcalEndcap/EELedClient/EELDT led timing L%(wl)s %(sm)s'),
            description = cms.untracked.string('Distribution of the timing in each crystal channel. Z scale is in LHC clocks. Channels with less than ' + str(minChannelEntries) + ' are not considered.')            
        ),
        PNQualitySummary = cms.untracked.PSet(
            path = cms.untracked.string('EcalEndcap/EESummaryClient/EELDT PN led quality summary L%(wl)s'),
            otype = cms.untracked.string('EEMEM'),
            multi = cms.untracked.PSet(
                wl = ecaldqmLedWavelengths
            ),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('Crystal'),
            description = cms.untracked.string('Summary of the led data quality in the PN diodes. A channel is red if mean / expected < ' + str(tolerancePNAmp) + ' or RMS / expected > ' + str(tolerancePNRMSRatio) + '. Expected amplitudes are ' + ('%.1f, %.1f' % tuple(expectedPNAmplitude)) + ' for led 1 and 2 respectively. Channels with less than ' + str(minChannelEntries) + ' are not considered.')
        ),
        TimingRMSMap = cms.untracked.PSet(
            path = cms.untracked.string('EcalEndcap/EELedClient/EELDT timing RMS L%(wl)s'),
            otype = cms.untracked.string('EE'),
            multi = cms.untracked.PSet(
                wl = ecaldqmLedWavelengths
            ),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('Crystal'),
            description = cms.untracked.string('2D distribution of the led timing RMS. Z scale is in LHC clocks. Channels with less than ' + str(minChannelEntries) + ' are not considered.')            
        ),
        AmplitudeMean = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            multi = cms.untracked.PSet(
                wl = ecaldqmLedWavelengths
            ),
            otype = cms.untracked.string('EESM'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(400.0),
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0.0)
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('EcalEndcap/EELedClient/EELDT amplitude L%(wl)s %(sm)s'),
            description = cms.untracked.string('Distribution of the mean amplitude seen in each crystal. Channels with less than ' + str(minChannelEntries) + ' are not considered.')            
        ),
        QualitySummary = cms.untracked.PSet(
            path = cms.untracked.string('EcalEndcap/EESummaryClient/EELDT%(suffix)s led quality summary L%(wl)s'),
            otype = cms.untracked.string('EE2P'),
            multi = cms.untracked.PSet(
                wl = ecaldqmLedWavelengths
            ),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('SuperCrystal'),
            description = cms.untracked.string('Summary of the led data quality. A channel is red either if mean / expected < ' + str(toleranceAmplitude) + ', or if RMS / expected > ' + str(toleranceAmpRMSRatio) + ', or if mean timing is off from expected by ' + str(toleranceTiming) + '. Expected amplitudes and timings are ' + ('%.1f, %.1f' % tuple(expectedAmplitude)) + ' and ' + ('%.1f, %.1f' % tuple(expectedTiming)) + ' for leds 1 and 2 respectively. Channels with less than ' + str(minChannelEntries) + ' are not considered.')
        ),
        Quality = cms.untracked.PSet(
            path = cms.untracked.string('EcalEndcap/EELedClient/EELDT led quality L%(wl)s %(sm)s'),
            otype = cms.untracked.string('EESM'),
            multi = cms.untracked.PSet(
                wl = ecaldqmLedWavelengths
            ),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('Crystal'),
            description = cms.untracked.string('Summary of the led data quality. A channel is red either if mean / expected < ' + str(toleranceAmplitude) + ', or if RMS / expected > ' + str(toleranceAmpRMSRatio) + ', or if mean timing is off from expected by ' + str(toleranceTiming) + '. Expected amplitudes and timings are ' + ('%.1f, %.1f' % tuple(expectedAmplitude)) + ' and ' + ('%.1f, %.1f' % tuple(expectedTiming)) + ' for leds 1 and 2 respectively. Channels with less than ' + str(minChannelEntries) + ' are not considered.')
        ),
        AmplitudeRMS = cms.untracked.PSet(
            path = cms.untracked.string('EcalEndcap/EELedClient/EELDT amplitude RMS L%(wl)s'),
            otype = cms.untracked.string('EE'),
            multi = cms.untracked.PSet(
                wl = ecaldqmLedWavelengths
            ),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('Crystal'),
            description = cms.untracked.string('2D distribution of the amplitude RMS. Channels with less than ' + str(minChannelEntries) + ' are not considered.')            
        )
    )
)
