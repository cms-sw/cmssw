EaxisEdges = []
for i in range(50) :
    EaxisEdges.append(pow(10., -0.5 + 2.5 / 50. * i))

timingTask = dict(
    energyThresholdEB = 1.,
    energyThresholdEE = 3.,
    MEs = dict(
        TimeMap = dict(path = "Timing/Profile/TimingTask timing", otype = 'SM', btype = 'Crystal', kind = 'TProfile2D', zaxis = {'low': -25., 'high': 25., 'title': 'time (ns)'}),
        TimeAll = dict(path = "Timing/TimingTask timing all 1D", otype = 'Ecal3P', btype = 'User', kind = 'TH1F', xaxis = {'nbins': 100, 'low': -25., 'high': 25., 'title': 'time (ns)'}),
        TimeAllMap = dict(path = "Timing/TimingTask timing all", otype = 'Ecal3P', btype = 'SuperCrystal', kind = 'TProfile2D', zaxis = {'low': -7., 'high': 7., 'title': 'time (ns)'}),
        TimeAmp = dict(path = "Timing/VsAmplitude/TimingTask timing v amplitude", otype = 'SM', btype = 'User', kind = 'TH2F', xaxis = {'edges': EaxisEdges, 'title': 'energy (GeV)'}, yaxis = {'nbins': 200, 'low': -50., 'high': 50., 'title': 'time (ns)'}),
        TimeAmpAll = dict(path = "Timing/TimingTask timing v amplitude all", otype = 'Ecal3P', btype = 'User', kind = 'TH2F', xaxis = {'edges': EaxisEdges, 'title': 'energy (GeV)'}, yaxis = {'nbins': 200, 'low': -50., 'high': 50., 'title': 'time (ns)'})
    )
)
