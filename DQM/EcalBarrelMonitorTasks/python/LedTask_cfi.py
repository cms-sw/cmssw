ecalLedTask = dict(
    emptyLSLimit = 3,
    MEs = dict(
        Amplitude = dict(path = "EcalEndcap/EELedTask/Led%(wl)s/EELDT amplitude %(sm)s L%(wl)s", otype = 'EESM', btype = 'Crystal', kind = 'TProfile2D', multi = 2),
        AmplitudeSummary = dict(path = "EcalEndcap/EELedTask/Led%(wl)s/EELDT amplitude map L%(wl)s%(suffix)s", otype = 'EE2P', btype = 'SuperCrystal', kind = 'TProfile2D', multi = 4),        
        Occupancy = dict(path = "EcalEndcap/EEOccupancyTask/EEOT led digi occupancy%(suffix)s", otype = 'EE2P', btype = 'Crystal', kind = 'TH2F'),
        Shape = dict(path = "EcalEndcap/EELedTask/Led%(wl)s/EELDT shape %(sm)s L%(wl)s", otype = 'EESM', btype = 'SuperCrystal', kind = 'TProfile2D', yaxis = {'nbins': 10, 'low': 0., 'high': 10.}, multi = 2),
        Timing = dict(path = "EcalEndcap/EELedTask/Led%(wl)s/EELDT timing %(sm)s L%(wl)s", otype = 'EESM', btype = 'Crystal', kind = 'TProfile2D', multi = 2),
        AOverP = dict(path = "EcalEndcap/EELedTask/Led%(wl)s/EELDT amplitude over PN %(sm)s L%(wl)s", otype = 'EESM', btype = 'Crystal', kind = 'TProfile2D', multi = 2),
        PNAmplitude = dict(path = "EcalEndcap/EELedTask/Led%(wl)s/PN/Gain16/EELDT PNs amplitude %(sm)s G16 L%(wl)s", otype = 'EESMMEM', btype = 'Crystal', kind = 'TProfile', multi = 2)
    )
)
