from DQM.EcalBarrelMonitorTasks.SelectiveReadoutTask_cfi import ecalSelectiveReadoutTask
from DQM.EcalBarrelMonitorTasks.TrigPrimTask_cfi import ecalTrigPrimTask

ecalSelectiveReadoutClient = dict(
    MEs = dict(
        FRDropped = dict(path = "%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT FR Flagged Dropped Readout%(suffix)s", otype = 'Ecal3P', btype = 'SuperCrystal', kind = 'TH2F', zaxis = {'title': 'rate'}),
        ZSReadout = dict(path = "%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT ZS Flagged Fully Readout%(suffix)s", otype = 'Ecal3P', btype = 'SuperCrystal', kind = 'TH2F', zaxis = {'title': 'rate'}),
        FR = dict(path = "%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT full readout SR Flags%(suffix)s", otype = 'Ecal3P', btype = 'SuperCrystal', kind = 'TH2F', zaxis = {'title': 'rate'}),
        RUForced = dict(path = "%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT readout unit with SR forced%(suffix)s", otype = 'Ecal3P', btype = 'SuperCrystal', kind = 'TH2F', zaxis = {'title': 'rate'}),
        ZS1 = dict(path = "%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT zero suppression 1 SR Flags%(suffix)s", otype = 'Ecal3P', btype = 'SuperCrystal', kind = 'TH2F', zaxis = {'title': 'rate'}),
        HighInterest = dict(path = "%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT high interest TT Flags%(suffix)s", otype = 'Ecal3P', btype = 'TriggerTower', kind = 'TH2F', zaxis = {'title': 'rate'}),
        MedInterest = dict(path = "%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT medium interest TT Flags%(suffix)s", otype = 'Ecal3P', btype = 'TriggerTower', kind = 'TH2F', zaxis = {'title': 'rate'}),
        LowInterest = dict(path = "%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT low interest TT Flags%(suffix)s", otype = 'Ecal3P', btype = 'TriggerTower', kind = 'TH2F', zaxis = {'title': 'rate'})
    ),
    sources = dict(
        FlagCounterMap = ecalSelectiveReadoutTask['MEs']['FlagCounterMap'],
        RUForcedMap = ecalSelectiveReadoutTask['MEs']['RUForcedMap'],
        FullReadoutMap = ecalSelectiveReadoutTask['MEs']['FullReadoutMap'],
        ZS1Map = ecalSelectiveReadoutTask['MEs']['ZS1Map'],
        ZSMap = ecalSelectiveReadoutTask['MEs']['ZSMap'],
        ZSFullReadoutMap = ecalSelectiveReadoutTask['MEs']['ZSFullReadoutMap'],
        FRDroppedMap = ecalSelectiveReadoutTask['MEs']['FRDroppedMap'],
        HighIntMap = ecalTrigPrimTask['MEs']['HighIntMap'],
        MedIntMap = ecalTrigPrimTask['MEs']['MedIntMap'],
        LowIntMap = ecalTrigPrimTask['MEs']['LowIntMap']
    )
)
