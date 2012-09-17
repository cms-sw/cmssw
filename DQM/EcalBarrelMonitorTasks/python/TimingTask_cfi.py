EaxisEdges = []
for i in range(50) :
    EaxisEdges.append(pow(10., -0.5 + 2.5 / 50. * i))

taxis = {'nbins': 100, 'low': -25., 'high': 25., 'title': 'time (ns)'}

ecalTimingTask = dict(
    energyThresholdEB = 1.,
    energyThresholdEE = 3.,
    MEs = dict(
        TimeMap = dict(path = "%(subdet)s/%(prefix)sTimingTask/%(prefix)sTMT timing %(sm)s", otype = 'SM', btype = 'Crystal', kind = 'TProfile2D', zaxis = {'low': -25., 'high': 25., 'title': 'time (ns)'}),
        Time1D = dict(path = "%(subdet)s/%(prefix)sTimingTask/%(prefix)sTMT timing 1D %(sm)s", otype = 'SM', btype = 'User', kind = 'TH1F', xaxis = taxis),
        TimeAll = dict(path = "%(subdet)s/%(prefix)sTimingTask/%(prefix)sTMT timing 1D summary%(suffix)s", otype = 'Ecal3P', btype = 'User', kind = 'TH1F', xaxis = taxis),
        TimeAllMap = dict(path = "%(subdet)s/%(prefix)sTimingTask/%(prefix)sTMT timing map%(suffix)s", otype = 'Ecal3P', btype = 'SuperCrystal', kind = 'TProfile2D', zaxis = {'low': -7., 'high': 7., 'title': 'time (ns)'}),
        TimeAmp = dict(path = "%(subdet)s/%(prefix)sTimingTask/%(prefix)sTMT timing vs amplitude %(sm)s", otype = 'SM', btype = 'User', kind = 'TH2F', xaxis = {'edges': EaxisEdges, 'title': 'energy (GeV)'}, yaxis = {'nbins': 200, 'low': -50., 'high': 50., 'title': 'time (ns)'}),
        TimeAmpAll = dict(path = "%(subdet)s/%(prefix)sTimingTask/%(prefix)sTMT timing vs amplitude summary%(suffix)s", otype = 'Ecal3P', btype = 'User', kind = 'TH2F', xaxis = {'edges': EaxisEdges, 'title': 'energy (GeV)'}, yaxis = {'nbins': 200, 'low': -50., 'high': 50., 'title': 'time (ns)'})
    )
)
