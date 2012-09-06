ledTask = dict(
    MEs = dict(
        AmplitudeSummary = dict(path = "Led/Led%(wl)s/LedTask amplitude summary L%(wl)s", otype = 'EE', btype = 'SuperCrystal', kind = 'TProfile2D', multi = 2),
        Amplitude = dict(path = "Led/Led%(wl)s/Amplitude/LedTask amplitude L%(wl)s", otype = 'EESM', btype = 'Crystal', kind = 'TProfile2D', multi = 2),
        Occupancy = dict(path = "Occupancy/LedTask digi occupancy L%(wl)s", otype = 'EE', btype = 'Crystal', kind = 'TH2F', multi = 2),
        Shape = dict(path = "Led/Led%(wl)s/Shape/LedTask pulse shape L%(wl)s", otype = 'EESM', btype = 'SuperCrystal', kind = 'TProfile2D', yaxis = {'nbins': 10, 'low': 0., 'high': 10.}, multi = 2),
        Timing = dict(path = "Led/Led%(wl)s/Timing/LedTask uncalib timing L%(wl)s", otype = 'EESM', btype = 'Crystal', kind = 'TProfile2D', multi = 2),
        AOverP = dict(path = "Led/Led%(wl)s/AOverP/LedTask AoverP L%(wl)s", otype = 'EESM', btype = 'Crystal', kind = 'TProfile2D', multi = 2),
        PNAmplitude = dict(path = "PN/Led/Led%(wl)s/LedTask PN amplitude L%(wl)s", otype = 'EESMMEM', btype = 'Crystal', kind = 'TProfile', multi = 2)
    )
)
