laserTask = dict(
    MEs = dict(
        AmplitudeSummary = dict(path = "Laser/Laser%(wl)s/LaserTask amplitude summary L%(wl)s", otype = 'Ecal2P', btype = 'SuperCrystal', kind = 'TProfile2D'),
        Amplitude = dict(path = "Laser/Laser%(wl)s/Amplitude/LaserTask amplitude L%(wl)s", otype = 'SM', btype = 'Crystal', kind = 'TProfile2D'),
        Occupancy = dict(path = "Occupancy/LaserTask digi occupancy L%(wl)s", otype = 'Ecal2P', btype = 'SuperCrystal', kind = 'TH2F'),
        Timing = dict(path = "Laser/Laser%(wl)s/Timing/LaserTask uncalib timing L%(wl)s", otype = 'SM', btype = 'Crystal', kind = 'TProfile2D'),
        Shape = dict(path = "Laser/Laser%(wl)s/Shape/LaserTask pulse shape L%(wl)s", otype = 'SM', btype = 'SuperCrystal', kind = 'TProfile2D', yaxis = {'nbins': 10, 'low': 0., 'high': 10.}),
        AOverP = dict(path = "Laser/Laser%(wl)s/AOverP/LaserTask AoverP L%(wl)s", otype = 'SM', btype = 'Crystal', kind = 'TProfile2D'),
        PNAmplitude = dict(path = "Laser/Laser%(wl)s/PN/Gain%(pngain)s/LaserTask PN amplitude L%(wl)s G%(pngain)s", otype = 'SMMEM', btype = 'Crystal', kind = 'TProfile'),
        PNOccupancy = dict(path = "Occupancy/LaserTask PN digi occupancy", otype = 'MEM', btype = 'Crystal', kind = 'TH2F')
    )
)
