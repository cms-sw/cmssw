import FWCore.ParameterSet.Config as cms

from DQM.EcalMonitorTasks.PresampleTask_cfi import ecalPresampleTask
from DQM.EcalMonitorClient.IntegrityClient_cfi import ecalIntegrityClient

minChannelEntries = 6
expectedMean = 200.0
toleranceLow = 25.0
toleranceHigh = 60.0
toleranceHighFwd = 100.0
toleranceRMS = 6.0
toleranceRMSFwd = 6.0

ecalPresampleClient = cms.untracked.PSet(
    params = cms.untracked.PSet(
        minChannelEntries = cms.untracked.int32(minChannelEntries),
        expectedMean = cms.untracked.double(expectedMean),
        toleranceLow = cms.untracked.double(toleranceLow),
        toleranceHigh = cms.untracked.double(toleranceHigh),
	toleranceHighFwd = cms.untracked.double(toleranceHighFwd),
        toleranceRMS = cms.untracked.double(toleranceRMS),
        toleranceRMSFwd = cms.untracked.double(toleranceRMSFwd)
    ),
    sources = cms.untracked.PSet(
        Pedestal = ecalPresampleTask.MEs.Pedestal,
        PedestalByLS = ecalPresampleTask.MEs.PedestalByLS,
        ChStatus = ecalIntegrityClient.MEs.ChStatus
    ),
    MEs = cms.untracked.PSet(
        RMS = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sPedestalOnlineClient/%(prefix)sPOT pedestal rms G12 %(sm)s'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('SM'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(10.0),
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0.0)
            ),
            btype = cms.untracked.string('User'),
            description = cms.untracked.string('Distribution of the presample RMS of each channel. Channels with entries less than ' + str(minChannelEntries) + ' are not considered.')
        ),
        TrendRMS = cms.untracked.PSet(
            path = cms.untracked.string('Ecal/Trends/PresampleClient %(prefix)s pedestal rms max'),
            kind = cms.untracked.string('TProfile'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('Trend'),
            description = cms.untracked.string('Trend of presample RMS averaged over all channels in EB / EE.')
        ),
        RMSMap = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sPedestalOnlineClient/%(prefix)sPOT pedestal rms map G12 %(sm)s'),
            kind = cms.untracked.string('TH2F'),
            zaxis = cms.untracked.PSet(
                title = cms.untracked.string('RMS')
            ),
            otype = cms.untracked.string('SM'),
            btype = cms.untracked.string('Crystal'),
            description = cms.untracked.string('2D distribution of the presample RMS. Channels with entries less than ' + str(minChannelEntries) + ' are not considered.')
        ),
        RMSMapAll = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sPOT%(suffix)s pedestal G12 RMS map'),
            kind = cms.untracked.string('TH2F'),
            zaxis = cms.untracked.PSet(
                title = cms.untracked.string('RMS')
            ),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('SuperCrystal'),
            description = cms.untracked.string('2D distribution of the presample RMS. Channels with entries less than ' + str(minChannelEntries) + ' are not considered.')
        ),
	 MeanMapAll = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sPOT%(suffix)s pedestal G12 Mean map'),
            kind = cms.untracked.string('TH2F'),
            zaxis = cms.untracked.PSet(
                title = cms.untracked.string('Mean')
            ),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('SuperCrystal'),
            description = cms.untracked.string('2D distribution of the presample Mean. Channels with entries less than ' + str(minChannelEntries) + ' are not considered.')
        ),
        RMSMapAllByLumi = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sPOT%(suffix)s pedestal G12 RMS map by lumi'),
            kind = cms.untracked.string('TH2F'),
            zaxis = cms.untracked.PSet(
                title = cms.untracked.string('RMS')
            ),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('Crystal'),
            description = cms.untracked.string('2D distribution of the presample RMS in this lumisection. Channels with entries less than ' + str(minChannelEntries) + ' are not considered.')
        ),
        TrendMean = cms.untracked.PSet(
            path = cms.untracked.string('Ecal/Trends/PresampleClient %(prefix)s pedestal mean max - min'),
            kind = cms.untracked.string('TProfile'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('Trend'),
            description = cms.untracked.string('Trend of presample spread in EB / EE. Y value indicates the difference between maximum and minimum presample mean values within the subdetector.')
        ),
        QualitySummary = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sPOT%(suffix)s pedestal quality summary G12'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('Crystal'),
            description = cms.untracked.string('Summary of the presample data quality. A channel is red if presample mean is outside the range (' + str(expectedMean - toleranceLow) + ', ' + str(expectedMean + toleranceHigh) + '), or (' + str(expectedMean - toleranceLow) + ', ' + str(expectedMean + toleranceHighFwd) + ') for forward region, or RMS is greater than ' + str(toleranceRMS) + '. RMS threshold is ' + str(toleranceRMSFwd) + ' in the forward region (|eta| > 2.1). Channels with entries less than ' + str(minChannelEntries) + ' are not considered.')
        ),
        Quality = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sPedestalOnlineClient/%(prefix)sPOT pedestal quality G12 %(sm)s'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('SM'),
            btype = cms.untracked.string('Crystal'),
            description = cms.untracked.string('Summary of the presample data quality. A channel is red if presample mean is outside the range (' + str(expectedMean - toleranceLow) + ', ' + str(expectedMean + toleranceHigh) + '), or (' + str(expectedMean - toleranceLow) + ', ' + str(expectedMean + toleranceHighFwd) + ') for forward region, or RMS is greater than ' + str(toleranceRMS) + '. RMS threshold is ' + str(toleranceRMSFwd) + ' in the forward region (|eta| > 2.1). Channels with entries less than ' + str(minChannelEntries) + ' are not considered.')
        ),
        ErrorsSummary = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sPOT pedestal quality errors summary G12'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('DCC'),
            description = cms.untracked.string('Counter of channels flagged as bad in the quality summary')
        ),
        Mean = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sPedestalOnlineClient/%(prefix)sPOT pedestal mean G12 %(sm)s'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('SM'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(270.0),
                nbins = cms.untracked.int32(200),
                low = cms.untracked.double(170.0)
            ),
            btype = cms.untracked.string('User'),
            description = cms.untracked.string('1D distribution of the mean presample value in each crystal. Channels with entries less than ' + str(minChannelEntries) + ' are not considered.')
        )
    )
)
