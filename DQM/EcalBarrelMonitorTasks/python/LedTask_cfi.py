ledTask = dict(
    ledWavelengths = [1, 2]
)

ledTaskPaths = dict(
    AmplitudeSummary = "Led/Led%(wl)s/LedTask amplitude summary L%(wl)s",
    Amplitude        = "Led/Led%(wl)s/Amplitude/LedTask amplitude L%(wl)s",
    Occupancy        = "Occupancy/LedTask digi occupancy L%(wl)s",
    Shape            = "Led/Led%(wl)s/LedTask pulse shape L%(wl)s",
    Timing           = "Led/Led%(wl)s/Timing/LedTask uncalib timing L%(wl)s",
    AOverP           = "Led/Led%(wl)s/AOverP/LedTask AoverP L%(wl)s",
    PNAmplitude      = "Led/Led%(wl)s/PN/Gain%(pngain)s/LedTask PN amplitude L%(wl)s G%(wl)s",
    PNOccupancy      = "Occupancy/LedTask PN digi occupancy"
)
