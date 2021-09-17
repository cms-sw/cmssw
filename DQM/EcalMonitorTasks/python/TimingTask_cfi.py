import FWCore.ParameterSet.Config as cms

bxBins = [1]

bxStepSizes = [9, 50, 100, 300]
bxMaxVals = [101, 1501, 2401, 3601]
runningMinVal = 1

for stepCounter in range(len(bxStepSizes)):
    runningMinVal = bxBins[-1]
    bxStepSize = bxStepSizes[stepCounter]
    bxMaxVal = bxMaxVals[stepCounter]
    bxBins += list(range(runningMinVal + bxStepSize, bxMaxVal, bxStepSize))

bxBinLabels = [str(bxBins[0])]
for bxBinCounter in range(0, -1+len(bxBins)):
    bxBinLabels += [str(1+bxBins[bxBinCounter]) + "-->" + str(bxBins[bxBinCounter+1])]

nBXBins = len(bxBins)

bxBinsFine = [i for i in range(1, 3601)]
bxBinLabelsFine = [str(i) for i in range(1, 3601)]
nBXBinsFine = len(bxBinsFine)

EaxisEdges = []
for i in range(50) :
    EaxisEdges.append(pow(10., -0.5 + 2.5 / 50. * i))

chi2ThresholdEE = 50.
chi2ThresholdEB = 16.
energyThresholdEE = 4.6
energyThresholdEEFwd = 6.7
energyThresholdEB = 2.02
timingVsBXThreshold = energyThresholdEB
timeErrorThreshold = 3.
timeWindow = 12.5
summaryTimeWindow = 7.

ecalTimingTask = cms.untracked.PSet(
    params = cms.untracked.PSet(
        bxBins = cms.untracked.vint32(bxBins),
        bxBinsFine = cms.untracked.vint32(bxBinsFine),
        chi2ThresholdEE = cms.untracked.double(chi2ThresholdEE),
        chi2ThresholdEB = cms.untracked.double(chi2ThresholdEB),
        energyThresholdEE = cms.untracked.double(energyThresholdEE),
        energyThresholdEEFwd = cms.untracked.double(energyThresholdEEFwd),
        energyThresholdEB = cms.untracked.double(energyThresholdEB),
        timingVsBXThreshold = cms.untracked.double(timingVsBXThreshold),
        timeErrorThreshold = cms.untracked.double(timeErrorThreshold)
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
        BarrelTimingVsBX = cms.untracked.PSet(
            path = cms.untracked.string('EcalBarrel/EBTimingTask/EBTMT Timing vs BX'),
            kind = cms.untracked.string('TProfile'),
            otype = cms.untracked.string('EB'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(1.0*nBXBins),
                nbins = cms.untracked.int32(nBXBins),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('BX Id'),
                labels = cms.untracked.vstring(bxBinLabels)
            ),
            yaxis = cms.untracked.PSet(
                title = cms.untracked.string('Timing (ns)')
            ),
            btype = cms.untracked.string('User'),
            description = cms.untracked.string('Average hit timing in EB as a function of BX number. BX ids start at 1. Only events with energy above 2.02 GeV and chi2 less than 16 are used.')
        ),
        BarrelTimingVsBXFineBinned = cms.untracked.PSet(
            path = cms.untracked.string('EcalBarrel/EBTimingTask/EBTMT Timing vs Finely Binned BX'),
            kind = cms.untracked.string('TProfile'),
            otype = cms.untracked.string('EB'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(1.0*nBXBinsFine),
                nbins = cms.untracked.int32(nBXBinsFine),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('BX Id'),
                labels = cms.untracked.vstring(bxBinLabelsFine)
            ),
            yaxis = cms.untracked.PSet(
                title = cms.untracked.string('Timing (ns)')
            ),
            btype = cms.untracked.string('User'),
            description = cms.untracked.string('Average hit timing in EB as a finely binned function of BX number. BX ids start at 1. Only events with energy above 2.02 GeV and chi2 less than 16 are used. The Customize button can be used to zoom in.')
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
