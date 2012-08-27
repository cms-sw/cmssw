from DQM.EcalBarrelMonitorTasks.SelectiveReadoutTask_cfi import selectiveReadoutTask

selectiveReadoutClient = dict(
    MEs = dict(
        FRDropped = dict(path = "SelectiveReadout/SRClient FR flag drop rate", otype = 'Ecal2P', btype = 'SuperCrystal', kind = 'TH2F', zaxis = {'title': 'rate'}),
        ZSReadout = dict(path = "SelectiveReadout/SRClient ZS flag readout rate", otype = 'Ecal2P', btype = 'SuperCrystal', kind = 'TH2F', zaxis = {'title': 'rate'}),
        FR = dict(path = "SelectiveReadout/SRClient FR flag rate", otype = 'Ecal2P', btype = 'SuperCrystal', kind = 'TH2F', zaxis = {'title': 'rate'}),
        RUForced = dict(path = "SelectiveReadout/SRClient unit forced readout rate", otype = 'Ecal2P', btype = 'SuperCrystal', kind = 'TH2F', zaxis = {'title': 'rate'}),
        ZS1 = dict(path = "SelectiveReadout/SRClient ZS1 flag rate", otype = 'Ecal2P', btype = 'SuperCrystal', kind = 'TH2F', zaxis = {'title': 'rate'})
    ),
    sources = dict(
        FlagCounterMap = selectiveReadoutTask['MEs']['FlagCounterMap'],
        RUForcedMap = selectiveReadoutTask['MEs']['RUForcedMap'],
        FullReadoutMap = selectiveReadoutTask['MEs']['FullReadoutMap'],
        ZS1Map = selectiveReadoutTask['MEs']['ZS1Map'],
        ZSMap = selectiveReadoutTask['MEs']['ZSMap'],
        ZSFullReadoutMap = selectiveReadoutTask['MEs']['ZSFullReadoutMap'],
        FRDroppedMap = selectiveReadoutTask['MEs']['FRDroppedMap']
    )
)
