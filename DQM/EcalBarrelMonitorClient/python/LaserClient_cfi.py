laserClient = dict(
    minChannelEntries = 3,
    expectedAmplitude = [1500.0, 1500.0, 1500.0, 1500.0],
    amplitudeThreshold = [1000.0, 1000.0, 1000.0, 1000.0],
    amplitudeRMSThreshold = [50.0, 50.0, 50.0, 50.0],
    expectedTiming = [5.5, 5.5, 5.5, 5.5],
    timingThreshold = [0.5, 0.5, 0.5, 0.5],
    timingRMSThreshold = [0.2, 0.2, 0.2, 0.2],
    expectedPNAmplitude = [800.0, 800.0, 800.0, 800.0, 800.0, 800.0, 800.0, 800.0],
    pnAmplitudeThreshold = [500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0],
    pnAmplitudeRMSThreshold = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
    towerThreshold = 0.1
)

laserClientPaths = dict(
    AmplitudeSummary = "Laser/Laser%(wl)s/LaserTask amplitude summary L%(wl)s",
    Amplitude        = "Laser/Laser%(wl)s/Amplitude/LaserTask amplitude L%(wl)s",
    Occupancy        = "Occupancy/LaserTask digi occupancy L%(wl)s",
    Timing           = "Laser/Laser%(wl)s/Timing/LaserTask uncalib timing L%(wl)s",
    AOverP           = "Laser/Laser%(wl)s/AOverP/LaserTask AoverP L%(wl)s",
    PNAmplitude      = "Laser/Laser%(wl)s/PN/Gain%(pngain)s/LaserTask PN amplitude L%(wl)s G%(wl)s",
    PNOccupancy      = "Occupancy/LaserTask PN digi occupancy"
)
