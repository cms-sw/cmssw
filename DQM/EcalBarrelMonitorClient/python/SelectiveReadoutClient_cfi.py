from DQM.EcalBarrelMonitorTasks.SelectiveReadoutTask_cfi import ecalSelectiveReadoutTask

ecalSelectiveReadoutClient = dict(
    MEs = dict(
        FRDropped = dict(path = "SelectiveReadout/SRClient FR flag drop rate", otype = 'Ecal2P', btype = 'SuperCrystal', kind = 'TH2F', zaxis = {'title': 'rate'}),
        ZSReadout = dict(path = "SelectiveReadout/SRClient ZS flag readout rate", otype = 'Ecal2P', btype = 'SuperCrystal', kind = 'TH2F', zaxis = {'title': 'rate'}),
        FR = dict(path = "SelectiveReadout/SRClient FR flag rate", otype = 'Ecal2P', btype = 'SuperCrystal', kind = 'TH2F', zaxis = {'title': 'rate'}),
        RUForced = dict(path = "SelectiveReadout/SRClient unit forced readout rate", otype = 'Ecal2P', btype = 'SuperCrystal', kind = 'TH2F', zaxis = {'title': 'rate'}),
        ZS1 = dict(path = "SelectiveReadout/SRClient ZS1 flag rate", otype = 'Ecal2P', btype = 'SuperCrystal', kind = 'TH2F', zaxis = {'title': 'rate'})
    ),
    sources = dict(
        FlagCounterMap = ecalSelectiveReadoutTask['MEs']['FlagCounterMap'],
        RUForcedMap = ecalSelectiveReadoutTask['MEs']['RUForcedMap'],
        FullReadoutMap = ecalSelectiveReadoutTask['MEs']['FullReadoutMap'],
        ZS1Map = ecalSelectiveReadoutTask['MEs']['ZS1Map'],
        ZSMap = ecalSelectiveReadoutTask['MEs']['ZSMap'],
        ZSFullReadoutMap = ecalSelectiveReadoutTask['MEs']['ZSFullReadoutMap'],
        FRDroppedMap = ecalSelectiveReadoutTask['MEs']['FRDroppedMap']
    )
)
