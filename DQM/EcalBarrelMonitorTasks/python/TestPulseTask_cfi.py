testPulseTask = dict(
    MEs = dict(
        Occupancy = dict(path = "Occupancy/TestPulseTask digi occupancy G%(gain)s", otype = 'Ecal2P', btype = 'SuperCrystal', kind = 'TH2F', multi = 3),
        Shape = dict(path = "TestPulse/Gain%(gain)s/Shape/TestPulseTask shape G%(gain)s", otype = 'SM', btype = 'SuperCrystal', kind = 'TProfile2D', yaxis = {'nbins': 10, 'low': 0., 'high': 10.}, multi = 3),
        Amplitude = dict(path = "TestPulse/Gain%(gain)s/Amplitude/TestPulseTask amplitude G%(gain)s", otype = 'SM', btype = 'Crystal', kind = 'TProfile2D', multi = 3),
        PNAmplitude = dict(path = "PN/TestPulse/Gain%(pngain)s/TestPulseTask PN amplitude G%(pngain)s", otype = 'SMMEM', btype = 'Crystal', kind = 'TProfile', multi = 2)
    )
)

