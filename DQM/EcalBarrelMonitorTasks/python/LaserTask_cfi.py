ecalLaserTask = dict(
    emptyLSLimit = 3,
    MEs = dict(
        Amplitude = dict(path = "%(subdet)s/%(prefix)sLaserTask/Laser%(wl)s/%(prefix)sLT amplitude %(sm)s L%(wl)s", otype = 'SM', btype = 'Crystal', kind = 'TProfile2D', multi = 4),
        AmplitudeSummary = dict(path = "%(subdet)s/%(prefix)sLaserTask/Laser%(wl)s/%(prefix)sLT amplitude map L%(wl)s%(suffix)s", otype = 'Ecal3P', btype = 'SuperCrystal', kind = 'TProfile2D', multi = 4),
        Occupancy = dict(path = "%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT laser digi occupancy%(suffix)s", otype = 'Ecal3P', btype = 'SuperCrystal', kind = 'TH2F'),
        Timing = dict(path = "%(subdet)s/%(prefix)sLaserTask/Laser%(wl)s/%(prefix)sLT timing %(sm)s L%(wl)s", otype = 'SM', btype = 'Crystal', kind = 'TProfile2D', multi = 4),
        Shape = dict(path = "%(subdet)s/%(prefix)sLaserTask/Laser%(wl)s/%(prefix)sLT shape %(sm)s L%(wl)s", otype = 'SM', btype = 'SuperCrystal', kind = 'TProfile2D', yaxis = {'nbins': 10, 'low': 0., 'high': 10.}, multi = 4),
        AOverP = dict(path = "%(subdet)s/%(prefix)sLaserTask/Laser%(wl)s/%(prefix)sLT amplitude over PN %(sm)s L%(wl)s", otype = 'SM', btype = 'Crystal', kind = 'TProfile2D', multi = 4),
        PNAmplitude = dict(path = "%(subdet)s/%(prefix)sLaserTask/Laser%(wl)s/PN/Gain16/%(prefix)sLT PNs amplitude %(sm)s G16 L%(wl)s", otype = 'SMMEM', btype = 'Crystal', kind = 'TProfile', multi = 4)
    )
)
