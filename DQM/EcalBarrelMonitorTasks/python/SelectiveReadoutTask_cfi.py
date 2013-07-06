selectiveReadoutTask = dict(
    useCondDb = False,
    DCCZS1stSample = 2,
    ZSFIRWeights = [-0.374, -0.374, -0.3629, 0.2721, 0.4681, 0.3707]
)

selectiveReadoutTaskPaths = dict(
    TowerSize        = "SelectiveReadout/SRTask tower event size",
    DCCSize          = "SelectiveReadout/SRTask DCC size",
    EventSize        = "SelectiveReadout/SRTask event size per DCC",
    FlagCounterMap   = "SelectiveReadout/Counters/SRTask tower flag counter",
    RUForcedMap      = "SelectiveReadout/Counters/SRTask RU with forced SR counter",
    FullReadoutMap   = "SelectiveReadout/Counters/SRTask tower full readout counter",
    FullReadout      = "SelectiveReadout/SRTask towers fully readout",
    ZS1Map           = "SelectiveReadout/Counters/SRTask tower ZS1 counter",
    ZSMap            = "SelectiveReadout/Counters/SRTask tower ZS1+ZS2 counter",
    ZSFullReadoutMap = "SelectiveReadout/Counters/SRTask ZS flagged full readout counter",
    ZSFullReadout    = "SelectiveReadout/SRTask towers ZS flagged fully readout",
    FRDroppedMap     = "SelectiveReadout/Counters/SRTask FR flagged dropped counter",
    FRDropped        = "SelectiveReadout/SRTask towers FR flagged dropped",
    HighIntPayload   = "SelectiveReadout/SRTask high interest payload per DCC",
    LowIntPayload    = "SelectiveReadout/SRTask low interest payload per DCC",
    HighIntOutput    = "SelectiveReadout/SRTask high interest filter output",
    LowIntOutput     = "SelectiveReadout/SRTask low interest filter output"
)
