selectiveReadoutClient = dict(
    sources = dict(
        FlagCounterMap = ['SelectiveReadoutTask', 'FlagCounterMap'],
        RUForcedMap = ['SelectiveReadoutTask', 'RUForcedMap'],
        FullReadoutMap = ['SelectiveReadoutTask', 'FullReadoutMap'],
        ZS1Map = ['SelectiveReadoutTask', 'ZS1Map'],
        ZSMap = ['SelectiveReadoutTask', 'ZSMap'],
        ZSFullReadoutMap = ['SelectiveReadoutTask', 'ZSFullReadoutMap'],
        FRDroppedMap = ['SelectiveReadoutTask', 'FRDroppedMap']
    )
)

selectiveReadoutClientPaths = dict(
    FRDropped = "SelectiveReadout/SRClient FR flag drop rate",
    ZSReadout = "SelectiveReadout/SRClient ZS flag readout rate",
    FR = "SelectiveReadout/SRClient FR flag rate",
    RUForced = "SelectiveReadout/SRClient unit forced readout rate",
    ZS1 = "SelectiveReadout/SRClient ZS1 flag rate"
)
