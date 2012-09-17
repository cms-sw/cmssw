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
        TowerSize = dict(path = "%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT tower event size%(suffix)s", otype = 'Ecal3P', btype = 'SuperCrystal', kind = 'TProfile2D', zaxis = {'title': 'size (bytes)'}),
        DCCSize = dict(path = "%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT event size vs DCC", otype = 'Ecal2P', btype = 'DCC', kind = 'TH2F', yaxis = {'edges': dccSizeBinEdges, 'title': 'event size (kB)'}),
        EventSize = dict(path = "%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT event size%(suffix)s", otype = 'Ecal3P', btype = 'User', kind = 'TH1F', xaxis = {'nbins': 100, 'low': 0., 'high': 3., 'title': 'event size (kB)'}),
        FlagCounterMap = dict(path = "%(subdet)s/%(prefix)sSelectiveReadoutTask/Counters/%(prefix)sSRT tower flag counter%(suffix)s", otype = 'Ecal3P', btype = 'SuperCrystal', kind = 'TH2F'),
        RUForcedMap = dict(path = "%(subdet)s/%(prefix)sSelectiveReadoutTask/Counters/%(prefix)sSRT RU with forced SR counter%(suffix)s", otype = 'Ecal3P', btype = 'SuperCrystal', kind = 'TH2F'),
        FullReadoutMap = dict(path = "%(subdet)s/%(prefix)sSelectiveReadoutTask/Counters/%(prefix)sSRT tower full readout counter%(suffix)s", otype = 'Ecal3P', btype = 'SuperCrystal', kind = 'TH2F'),
        FullReadout = dict(path = "%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT full readout SR Flags Number%(suffix)s", otype = 'Ecal3P', btype = 'User', kind = 'TH1F', xaxis = {'nbins': 100, 'low': 0., 'high': 200., 'title': 'number of towers'}),
        ZS1Map = dict(path = "%(subdet)s/%(prefix)sSelectiveReadoutTask/Counters/%(prefix)sSRT tower ZS1 counter%(suffix)s", otype = 'Ecal3P', btype = 'SuperCrystal', kind = 'TH2F'),
        ZSMap = dict(path = "%(subdet)s/%(prefix)sSelectiveReadoutTask/Counters/%(prefix)sSRT tower ZS1+ZS2 counter%(suffix)s", otype = 'Ecal3P', btype = 'SuperCrystal', kind = 'TH2F'),
        ZSFullReadoutMap = dict(path = "%(subdet)s/%(prefix)sSelectiveReadoutTask/Counters/%(prefix)sSRT ZS flagged full readout counter%(suffix)s", otype = 'Ecal3P', btype = 'SuperCrystal', kind = 'TH2F'),
        ZSFullReadout = dict(path = "%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT ZS Flagged Fully Readout Number%(suffix)s", otype = 'Ecal3P', btype = 'User', kind = 'TH1F', xaxis = {'nbins': 20, 'low': 0., 'high': 20., 'title': 'number of towers'}),
        FRDroppedMap = dict(path = "%(subdet)s/%(prefix)sSelectiveReadoutTask/Counters/%(prefix)sSRT FR flagged dropped counter%(suffix)s", otype = 'Ecal3P', btype = 'SuperCrystal', kind = 'TH2F'),
        FRDropped = dict(path = "%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT FR Flagged Dropped Readout Number%(suffix)s", otype = 'Ecal3P', btype = 'User', kind = 'TH1F', xaxis = {'nbins': 20, 'low': 0., 'high': 20., 'title': 'number of towers'}),
        HighIntPayload = dict(path = "%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT high interest payload%(suffix)s", otype = 'Ecal3P', btype = 'User', kind = 'TH1F', xaxis = {'nbins': 100, 'low': 0., 'high': 3., 'title': 'event size (kB)'}),
        LowIntPayload = dict(path = "%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT low interest payload%(suffix)s", otype = 'Ecal3P', btype = 'User', kind = 'TH1F', xaxis = {'nbins': 100, 'low': 0., 'high': 3., 'title': 'event size (kB)'}),
        HighIntOutput = dict(path = "%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT high interest ZS filter output%(suffix)s", otype = 'Ecal3P', btype = 'User', kind = 'TH1F', xaxis = {'nbins': 100, 'low': -60., 'high': 60., 'title': 'ADC counts*4'}),
        LowIntOutput = dict(path = "%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT low interest ZS filter output%(suffix)s", otype = 'Ecal3P', btype = 'User', kind = 'TH1F', xaxis = {'nbins': 100, 'low': -60., 'high': 60., 'title': 'ADC counts*4'})
    )
)
