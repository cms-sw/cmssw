ledTask = dict(
    MEs = dict(
        AmplitudeSummary = dict(path = "Led/Led%(wl)s/LedTask amplitude summary L%(wl)s", otype = 'EE', btype = 'SuperCrystal', kind = 'TProfile2D'),
        Amplitude = dict(path = "Led/Led%(wl)s/Amplitude/LedTask amplitude L%(wl)s", otype = 'EESM', btype = 'SuperCrystal', kind = 'TProfile2D'),
        Occupancy = dict(path = "Occupancy/LedTask digi occupancy L%(wl)s", otype = 'EE', btype = 'SuperCrystal', kind = 'TH2F'),
        Shape = dict(path = "Led/Led%(wl)s/LedTask pulse shape L%(wl)s", otype = 'EESM', btype = 'SuperCrystal', kind = 'TProfile2D', yaxis = {'nbins': 10, 'low': 0., 'high': 10.}),
        Timing = dict(path = "Led/Led%(wl)s/Timing/LedTask uncalib timing L%(wl)s", otype = 'EESM', btype = 'SuperCrystal', kind = 'TProfile2D'),
        AOverP = dict(path = "Led/Led%(wl)s/AOverP/LedTask AoverP L%(wl)s", otype = 'EESM', btype = 'SuperCrystal', kind = 'TProfile2D'),
        PNAmplitude = dict(path = "Led/Led%(wl)s/PN/Gain%(pngain)s/LedTask PN amplitude L%(wl)s G%(wl)s", otype = 'EESMMEM', btype = 'Crystal', kind = 'TProfile'),
        PNOccupancy = dict(path = "Occupancy/LedTask PN digi occupancy", otype = 'EEMEM', btype = 'Crystal', kind = 'TH2F')
    )
)
