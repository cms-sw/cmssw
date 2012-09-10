laserTask = dict(
    emptyLSLimit = 3,
    MEs = dict(
        AmplitudeSummary = dict(path = "Laser/Laser%(wl)s/LaserTask amplitude summary L%(wl)s", otype = 'Ecal2P', btype = 'SuperCrystal', kind = 'TProfile2D', multi = 4),
        Amplitude = dict(path = "Laser/Laser%(wl)s/Amplitude/LaserTask amplitude L%(wl)s", otype = 'SM', btype = 'Crystal', kind = 'TProfile2D', multi = 4),
        Occupancy = dict(path = "Occupancy/LaserTask digi occupancy L%(wl)s", otype = 'Ecal2P', btype = 'SuperCrystal', kind = 'TH2F', multi = 4),
        Timing = dict(path = "Laser/Laser%(wl)s/Timing/LaserTask uncalib timing L%(wl)s", otype = 'SM', btype = 'Crystal', kind = 'TProfile2D', multi = 4),
        Shape = dict(path = "Laser/Laser%(wl)s/Shape/LaserTask pulse shape L%(wl)s", otype = 'SM', btype = 'SuperCrystal', kind = 'TProfile2D', yaxis = {'nbins': 10, 'low': 0., 'high': 10.}, multi = 4),
        AOverP = dict(path = "Laser/Laser%(wl)s/AOverP/LaserTask AoverP L%(wl)s", otype = 'SM', btype = 'Crystal', kind = 'TProfile2D', multi = 4),
        PNAmplitude = dict(path = "PN/Laser/Laser%(wl)s/LaserTask PN amplitude L%(wl)s", otype = 'SMMEM', btype = 'Crystal', kind = 'TProfile', multi = 4)
    )
)
