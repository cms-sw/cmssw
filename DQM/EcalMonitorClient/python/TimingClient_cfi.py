import FWCore.ParameterSet.Config as cms

from DQM.EcalMonitorTasks.TimingTask_cfi import ecalTimingTask

minChannelEntries = 5
minTowerEntries = 15
toleranceMean = 2.
toleranceRMS = 6.
minChannelEntriesFwd = 40
minTowerEntriesFwd = 160
toleranceMeanFwd = 6.
toleranceRMSFwd = 12.
tailPopulThreshold = 0.4

ecalTimingClient = cms.untracked.PSet(
    params = cms.untracked.PSet(
        minChannelEntries = cms.untracked.int32(minChannelEntries),
        minTowerEntries = cms.untracked.int32(minTowerEntries),
        toleranceMean = cms.untracked.double(toleranceMean),
        toleranceRMS = cms.untracked.double(toleranceRMS),
        minChannelEntriesFwd = cms.untracked.int32(minChannelEntriesFwd),
        minTowerEntriesFwd = cms.untracked.int32(minTowerEntriesFwd),
        toleranceMeanFwd = cms.untracked.double(toleranceMeanFwd),
        toleranceRMSFwd = cms.untracked.double(toleranceRMSFwd),
        tailPopulThreshold = cms.untracked.double(tailPopulThreshold)
    ),
    sources = cms.untracked.PSet(
        TimeAllMap = ecalTimingTask.MEs.TimeAllMap,
        TimeMap = ecalTimingTask.MEs.TimeMap
    ),
    MEs = cms.untracked.PSet(
        RMSAll = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sTMT%(suffix)s timing rms 1D summary'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal3P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(10.0),
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('time (ns)')
            ),
            btype = cms.untracked.string('User'),
            description = cms.untracked.string('Distribution of per-channel timing RMS. Channels with entries less than ' + str(minChannelEntries) + ' are not considered.')
        ),
        ProjEta = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sTimingClient/%(prefix)sTMT timing projection eta%(suffix)s'),
            kind = cms.untracked.string('TProfile'),
            yaxis = cms.untracked.PSet(
                title = cms.untracked.string('time (ns)')
            ),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('ProjEta'),
            description = cms.untracked.string('Projection of per-channel mean timing. Channels with entries less than ' + str(minChannelEntries) + ' are not considered.')
        ),
        FwdBkwdDiff = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sTimingTask/%(prefix)sTMT timing %(prefix)s+ - %(prefix)s-'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(5.0),
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(-5.0),
                title = cms.untracked.string('time (ns)')
            ),
            btype = cms.untracked.string('User'),
            description = cms.untracked.string('Forward-backward asymmetry of per-channel mean timing. Channels with entries less than ' + str(minChannelEntries) + ' are not considered.')
        ),
        FwdvBkwd = cms.untracked.PSet(
            kind = cms.untracked.string('TH2F'),
            yaxis = cms.untracked.PSet(
                high = cms.untracked.double(25.0),
                nbins = cms.untracked.int32(50),
                low = cms.untracked.double(-25.0),
                title = cms.untracked.string('time (ns)')
            ),
            otype = cms.untracked.string('Ecal2P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(25.0),
                nbins = cms.untracked.int32(50),
                low = cms.untracked.double(-25.0)
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sTimingTask/%(prefix)sTMT timing %(prefix)s+ vs %(prefix)s-'),
            description = cms.untracked.string('Forward-backward correlation of per-channel mean timing. Channels with entries less than ' + str(minChannelEntries) + ' are not considered.')
        ),
        ProjPhi = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sTimingClient/%(prefix)sTMT timing projection phi%(suffix)s'),
            kind = cms.untracked.string('TProfile'),
            yaxis = cms.untracked.PSet(
                title = cms.untracked.string('time (ns)')
            ),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('ProjPhi'),
            description = cms.untracked.string('Projection of per-channel mean timing. Channels with entries less than ' + str(minChannelEntries) + ' are not considered.')
        ),
        MeanSM = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            yaxis = cms.untracked.PSet(
                title = cms.untracked.string('time (ns)')
            ),
            otype = cms.untracked.string('SM'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(25.0),
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(-25.0)
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sTimingClient/%(prefix)sTMT timing mean %(sm)s'),
            description = cms.untracked.string('Distribution of per-channel timing mean. Channels with entries less than ' + str(minChannelEntries) + ' are not considered.')
        ),
        RMSMap = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sTimingClient/%(prefix)sTMT timing rms %(sm)s'),
            kind = cms.untracked.string('TH2F'),
            zaxis = cms.untracked.PSet(
                title = cms.untracked.string('rms (ns)')
            ),
            otype = cms.untracked.string('SM'),
            btype = cms.untracked.string('Crystal'),
            description = cms.untracked.string('2D distribution of per-channel timing RMS. Channels with entries less than ' + str(minChannelEntries) + ' are not considered.')
        ),
        QualitySummary = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sTMT%(suffix)s timing quality summary'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('SuperCrystal'),
            description = cms.untracked.string('Summary of the timing data quality. A 5x5 tower is red if the mean timing of the tower is off by more than ' + str(toleranceMean) + ' or RMS is greater than ' + str(toleranceRMS) + ' (' + str(toleranceMeanFwd) + ' and ' + str(toleranceRMSFwd) + ' in forward region). Towers with total entries less than ' + str(minTowerEntries) + ' are not subject to this evaluation. Since 5x5 tower timings are calculated with a tighter time-window than per-channel timings, a tower can additionally become red if its the sum of per-channel timing histogram entries is greater than per-tower histogram entries by factor ' + str(1. / (1. - tailPopulThreshold)) + ' (significant fraction of events fall outside the tight time-window).')
        ),
        Quality = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sTimingClient/%(prefix)sTMT timing quality %(sm)s'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('SM'),
            btype = cms.untracked.string('Crystal'),
            description = cms.untracked.string('Summary of the timing data quality. A channel is red if its mean timing is off by more than ' + str(toleranceMean) + ' or RMS is greater than ' + str(toleranceRMS) + '. Channels with entries less than ' + str(minChannelEntries) + ' are not considered.')
        ),
        MeanAll = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sTMT%(suffix)s timing mean 1D summary'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal3P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(25.0),
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(-25.0),
                title = cms.untracked.string('time (ns)')
            ),
            btype = cms.untracked.string('User'),
            description = cms.untracked.string('Distribution of per-channel timing mean. Channels with entries less than ' + str(minChannelEntries) + ' are not considered.')
        )
    )
)
