import FWCore.ParameterSet.Config as cms

EaxisEdges = []
for i in range(50) :
    EaxisEdges.append(pow(10., -0.5 + 2.5 / 50. * i))

chi2ThresholdEE = 50.
chi2ThresholdEB = 16.
energyThresholdEE = 3.
energyThresholdEB = 1.
timeWindow = 12.5
summaryTimeWindow = 7.

ecalTimingTask = cms.untracked.PSet(
    params = cms.untracked.PSet(
        chi2ThresholdEE = cms.untracked.double(chi2ThresholdEE),
        chi2ThresholdEB = cms.untracked.double(chi2ThresholdEB),
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
        TimeMapByLS = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sTimingTask/%(prefix)sTMT timing by LS %(sm)s'),
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
        TimeAmpBXm = cms.untracked.PSet(
            kind = cms.untracked.string('TH2F'),
            yaxis = cms.untracked.PSet(
                high = cms.untracked.double(100.0),
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('Amplitude BX-1 [ADC]')
            ),
            otype = cms.untracked.string('Ecal3P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(1000.0),
                nbins = cms.untracked.int32(250),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('In-time amplitude [ADC]')
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sTimingTask/%(prefix)sTMT in-time vs BX-1 amplitude%(suffix)s'),
            description = cms.untracked.string('Correlation between in-time amplitude and BX-1 out-of-time amplitude. Only events with kGood reconstruction flag set, energy > ( ' + ('EB:%f, EE:%f' % (energyThresholdEB*20., energyThresholdEE*5.)) + ' ) GeV, and chi2 < ( ' + ('EB:%f, EE:%f' % (chi2ThresholdEB, chi2ThresholdEE)) + ' ) are used.')
        ),
        TimeAmpBXp = cms.untracked.PSet(
            kind = cms.untracked.string('TH2F'),
            yaxis = cms.untracked.PSet(
                high = cms.untracked.double(100.0),
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('Amplitude BX+1 [ADC]')
            ),
            otype = cms.untracked.string('Ecal3P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(1000.0),
                nbins = cms.untracked.int32(250),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('In-time amplitude [ADC]')
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sTimingTask/%(prefix)sTMT in-time vs BX+1 amplitude%(suffix)s'),
            description = cms.untracked.string('Correlation between in-time amplitude and BX+1 out-of-time amplitude. Only events with kGood reconstruction flag set, energy > ( ' + ('EB:%f, EE:%f' % (energyThresholdEB*20., energyThresholdEE*5.)) + ' ) GeV, and chi2 < ( ' + ('EB:%f, EE:%f' % (chi2ThresholdEB, chi2ThresholdEE)) + ' ) are used.')
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
        ),
        Chi2 = cms.untracked.PSet(
            path = cms.untracked.string("%(subdet)s/%(prefix)sTimingTask/%(prefix)sTMT %(subdetshortsig)s Chi2"),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('User'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(100.),
                low = cms.untracked.double(0.),
                nbins = cms.untracked.int32(100)
            ),
            description = cms.untracked.string('Chi2 of the pulse reconstruction.')
        )
    )
)
