dccSizeBinEdges = []
for i in range(11) :
    dccSizeBinEdges.append(0.608 / 10. * i)
for i in range(11, 79) :
    dccSizeBinEdges.append(0.608 * (i - 10.))

ecalSelectiveReadoutTask = dict(
    useCondDb = False,
    DCCZS1stSample = 2,
    ZSFIRWeights = [-0.374, -0.374, -0.3629, 0.2721, 0.4681, 0.3707],
    MEs = dict(
        TowerSize = dict(path = "SelectiveReadout/SRTask tower event size", otype = 'Ecal2P', btype = 'SuperCrystal', kind = 'TProfile2D', zaxis = {'title': 'size (bytes)'}),
        DCCSize = dict(path = "SelectiveReadout/SRTask DCC size", otype = 'Ecal2P', btype = 'DCC', kind = 'TH2F', yaxis = {'edges': dccSizeBinEdges, 'title': 'event size (kB)'}),
        EventSize = dict(path = "SelectiveReadout/SRTask event size per DCC", otype = 'Ecal2P', btype = 'User', kind = 'TH1F', xaxis = {'nbins': 100, 'low': 0., 'high': 3., 'title': 'event size (kB)'}),
        FlagCounterMap = dict(path = "SelectiveReadout/Counters/SRTask tower flag counter", otype = 'Ecal2P', btype = 'SuperCrystal', kind = 'TH2F'),
        RUForcedMap = dict(path = "SelectiveReadout/Counters/SRTask RU with forced SR counter", otype = 'Ecal2P', btype = 'SuperCrystal', kind = 'TH2F'),
        FullReadoutMap = dict(path = "SelectiveReadout/Counters/SRTask tower full readout counter", otype = 'Ecal2P', btype = 'SuperCrystal', kind = 'TH2F'),
        FullReadout = dict(path = "SelectiveReadout/SRTask towers fully readout", otype = 'Ecal2P', btype = 'User', kind = 'TH1F', xaxis = {'nbins': 100, 'low': 0., 'high': 200., 'title': 'number of towers'}),
        ZS1Map = dict(path = "SelectiveReadout/Counters/SRTask tower ZS1 counter", otype = 'Ecal2P', btype = 'SuperCrystal', kind = 'TH2F'),
        ZSMap = dict(path = "SelectiveReadout/Counters/SRTask tower ZS1+ZS2 counter", otype = 'Ecal2P', btype = 'SuperCrystal', kind = 'TH2F'),
        ZSFullReadoutMap = dict(path = "SelectiveReadout/Counters/SRTask ZS flagged full readout counter", otype = 'Ecal2P', btype = 'SuperCrystal', kind = 'TH2F'),
        ZSFullReadout = dict(path = "SelectiveReadout/SRTask towers ZS flagged fully readout", otype = 'Ecal2P', btype = 'User', kind = 'TH1F', xaxis = {'nbins': 20, 'low': 0., 'high': 20., 'title': 'number of towers'}),
        FRDroppedMap = dict(path = "SelectiveReadout/Counters/SRTask FR flagged dropped counter", otype = 'Ecal2P', btype = 'SuperCrystal', kind = 'TH2F'),
        FRDropped = dict(path = "SelectiveReadout/SRTask towers FR flagged dropped", otype = 'Ecal2P', btype = 'User', kind = 'TH1F', xaxis = {'nbins': 20, 'low': 0., 'high': 20., 'title': 'number of towers'}),
        HighIntPayload = dict(path = "SelectiveReadout/SRTask high interest payload per DCC", otype = 'Ecal2P', btype = 'User', kind = 'TH1F', xaxis = {'nbins': 100, 'low': 0., 'high': 3., 'title': 'event size (kB)'}),
        LowIntPayload = dict(path = "SelectiveReadout/SRTask low interest payload per DCC", otype = 'Ecal2P', btype = 'User', kind = 'TH1F', xaxis = {'nbins': 100, 'low': 0., 'high': 3., 'title': 'event size (kB)'}),
        HighIntOutput = dict(path = "SelectiveReadout/SRTask high interest filter output", otype = 'Ecal2P', btype = 'User', kind = 'TH1F', xaxis = {'nbins': 100, 'low': -60., 'high': 60., 'title': 'ADC counts*4'}),
        LowIntOutput = dict(path = "SelectiveReadout/SRTask low interest filter output", otype = 'Ecal2P', btype = 'User', kind = 'TH1F', xaxis = {'nbins': 100, 'low': -60., 'high': 60., 'title': 'ADC counts*4'})
    )
)
