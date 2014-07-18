import FWCore.ParameterSet.Config as cms

EaxisEdges = []
for i in range(50) :
    EaxisEdges.append(pow(10., -0.5 + 2.5 / 50. * i))

energyThresholdEE = 3.
energyThresholdEB = 1.
timeWindow = 25.
summaryTimeWindow = 7.

ecalTimingTask = cms.untracked.PSet(
    params = cms.untracked.PSet(
        energyThresholdEE = cms.untracked.double(energyThresholdEE),
        energyThresholdEB = cms.untracked.double(energyThresholdEB)
    ),
    MEs = cms.untracked.PSet(
        TimeMap = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sTimingTask/%(prefix)sTMT timing %(sm)s'),
            kind = cms.untracked.string('TProfile2D'),
            zaxis = cms.untracked.PSet(
                high = cms.untracked.double(timeWindow),
                low = cms.untracked.double(-timeWindow),
                title = cms.untracked.string('time (ns)')
            ),
            otype = cms.untracked.string('SM'),
            btype = cms.untracked.string('Crystal'),
            description = cms.untracked.string('2D distribution of the mean rec hit timing. Only hits with GOOD or OUT_OF_TIME reconstruction flags and energy above threshold are used. Hits with |t| > ' + str(timeWindow) + ' ns are discarded. The energy thresholds are ' + ('%f and %f' % (energyThresholdEB, energyThresholdEE)) + ' for EB and EE respectively.')
        ),
        TimeAll = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sTimingTask/%(prefix)sTMT timing 1D summary%(suffix)s'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal3P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(timeWindow),
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(-timeWindow),
                title = cms.untracked.string('time (ns)')
            ),
            btype = cms.untracked.string('User'),
            description = cms.untracked.string('Distribution of the mean rec hit timing. Only hits with GOOD or OUT_OF_TIME reconstruction flags and energy above threshold are used. The energy thresholds are ' + ('%f and %f' % (energyThresholdEB, energyThresholdEE)) + ' for EB and EE respectively.')
        ),
        TimeAllMap = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sTimingTask/%(prefix)sTMT timing map%(suffix)s'),
            kind = cms.untracked.string('TProfile2D'),
            zaxis = cms.untracked.PSet(
                high = cms.untracked.double(summaryTimeWindow),
                low = cms.untracked.double(-summaryTimeWindow),
                title = cms.untracked.string('time (ns)')
            ),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('SuperCrystal'),
            description = cms.untracked.string('2D distribution of the mean rec hit timing. Only hits with GOOD or OUT_OF_TIME reconstruction flags and energy above threshold are used. Hits with |t| > ' + str(summaryTimeWindow) + ' ns are discarded. The energy thresholds are ' + ('%f and %f' % (energyThresholdEB, energyThresholdEE)) + ' for EB and EE respectively.')
        ),
        TimeAmpAll = cms.untracked.PSet(
            kind = cms.untracked.string('TH2F'),
            yaxis = cms.untracked.PSet(
                high = cms.untracked.double(50.0),
                nbins = cms.untracked.int32(200),
                low = cms.untracked.double(-50.0),
                title = cms.untracked.string('time (ns)')
            ),
            otype = cms.untracked.string('Ecal3P'),
            xaxis = cms.untracked.PSet(
                edges = cms.untracked.vdouble(EaxisEdges),
                title = cms.untracked.string('energy (GeV)')
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sTimingTask/%(prefix)sTMT timing vs amplitude summary%(suffix)s'),
            description = cms.untracked.string('Correlation between hit timing and energy. Only hits with GOOD or OUT_OF_TIME reconstruction flags are used.')
        ),
        TimeAmp = cms.untracked.PSet(
            kind = cms.untracked.string('TH2F'),
            yaxis = cms.untracked.PSet(
                high = cms.untracked.double(50.0),
                nbins = cms.untracked.int32(200),
                low = cms.untracked.double(-50.0),
                title = cms.untracked.string('time (ns)')
            ),
            otype = cms.untracked.string('SM'),
            xaxis = cms.untracked.PSet(
                edges = cms.untracked.vdouble(EaxisEdges),
                title = cms.untracked.string('energy (GeV)')
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sTimingTask/%(prefix)sTMT timing vs amplitude %(sm)s'),
            description = cms.untracked.string('Correlation between hit timing and energy. Only hits with GOOD or OUT_OF_TIME reconstruction flags are used.')
        ),
        Time1D = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sTimingTask/%(prefix)sTMT timing 1D %(sm)s'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('SM'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(timeWindow),
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(-timeWindow),
                title = cms.untracked.string('time (ns)')
            ),
            btype = cms.untracked.string('User'),
            description = cms.untracked.string('Distribution of the mean rec hit timing. Only hits with GOOD or OUT_OF_TIME reconstruction flags and energy above threshold are used. The energy thresholds are ' + ('%f and %f' % (energyThresholdEB, energyThresholdEE)) + ' for EB and EE respectively.')
        )
    )
)
