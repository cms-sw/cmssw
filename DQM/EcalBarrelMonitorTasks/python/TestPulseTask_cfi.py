ecalTestPulseTask = dict(
    MEs = dict(
        Occupancy = dict(path = "%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT test pulse digi occupancy%(suffix)s", otype = 'Ecal3P', btype = 'SuperCrystal', kind = 'TH2F'),
        Shape = dict(path = "%(subdet)s/%(prefix)sTestPulseTask/Gain%(gain)s/%(prefix)sTPT shape %(sm)s G%(gain)s", otype = 'SM', btype = 'SuperCrystal', kind = 'TProfile2D', yaxis = {'nbins': 10, 'low': 0., 'high': 10.}, multi = 3),
        Amplitude = dict(path = "%(subdet)s/%(prefix)sTestPulseTask/Gain%(gain)s/%(prefix)sTPT amplitude %(sm)s G%(gain)s", otype = 'SM', btype = 'Crystal', kind = 'TProfile2D', multi = 3),
        PNAmplitude = dict(path = "%(subdet)s/%(prefix)sTestPulseTask/PN/Gain%(pngain)s/%(prefix)sTPT PNs amplitude %(sm)s G%(pngain)s", otype = 'SMMEM', btype = 'Crystal', kind = 'TProfile', multi = 2)
    )
)

