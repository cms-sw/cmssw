import FWCore.ParameterSet.Config as cms

from DQM.EcalMonitorTasks.SelectiveReadoutTask_cfi import ecalSelectiveReadoutTask
from DQM.EcalMonitorTasks.TrigPrimTask_cfi import ecalTrigPrimTask

ecalSelectiveReadoutClient = cms.untracked.PSet(
    sources = cms.untracked.PSet(
        FlagCounterMap = ecalSelectiveReadoutTask.MEs.FlagCounterMap,
        RUForcedMap = ecalSelectiveReadoutTask.MEs.RUForcedMap,
        FullReadoutMap = ecalSelectiveReadoutTask.MEs.FullReadoutMap,
        ZS1Map = ecalSelectiveReadoutTask.MEs.ZS1Map,
        ZSMap = ecalSelectiveReadoutTask.MEs.ZSMap,
        ZSFullReadoutMap = ecalSelectiveReadoutTask.MEs.ZSFullReadoutMap,
        FRDroppedMap = ecalSelectiveReadoutTask.MEs.FRDroppedMap,
        HighIntMap = ecalTrigPrimTask.MEs.HighIntMap,
        MedIntMap = ecalTrigPrimTask.MEs.MedIntMap,
        LowIntMap = ecalTrigPrimTask.MEs.LowIntMap        
    ),
    MEs = cms.untracked.PSet(
        FR = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT full readout SR Flags%(suffix)s'),
            kind = cms.untracked.string('TH2F'),
            zaxis = cms.untracked.PSet(
                title = cms.untracked.string('rate')
            ),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('SuperCrystal'),
            description = cms.untracked.string('Occurrence rate of full readout flag.')
        ),
        LowInterest = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT low interest TT Flags%(suffix)s'),
            kind = cms.untracked.string('TH2F'),
            zaxis = cms.untracked.PSet(
                title = cms.untracked.string('rate')
            ),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('TriggerTower'),
            description = cms.untracked.string('Occurrence rate of low interest TT flags.')
        ),
        RUForced = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT readout unit with SR forced%(suffix)s'),
            kind = cms.untracked.string('TH2F'),
            zaxis = cms.untracked.PSet(
                title = cms.untracked.string('rate')
            ),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('SuperCrystal'),
            description = cms.untracked.string('Occurrence rate of forced selective readout.')
        ),
        ZS1 = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT zero suppression 1 SR Flags%(suffix)s'),
            kind = cms.untracked.string('TH2F'),
            zaxis = cms.untracked.PSet(
                title = cms.untracked.string('rate')
            ),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('SuperCrystal'),
            description = cms.untracked.string('Occurrence rate of zero suppression 1 flags.')
        ),
        MedInterest = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT medium interest TT Flags%(suffix)s'),
            kind = cms.untracked.string('TH2F'),
            zaxis = cms.untracked.PSet(
                title = cms.untracked.string('rate')
            ),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('TriggerTower'),
            description = cms.untracked.string('Occurrence rate of medium interest TT flags.')
        ),
        HighInterest = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT high interest TT Flags%(suffix)s'),
            kind = cms.untracked.string('TH2F'),
            zaxis = cms.untracked.PSet(
                title = cms.untracked.string('rate')
            ),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('TriggerTower'),
            description = cms.untracked.string('Occurrence rate of high interest TT flags.')
        ),
        ZSReadout = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT ZS Flagged Fully Readout%(suffix)s'),
            kind = cms.untracked.string('TH2F'),
            zaxis = cms.untracked.PSet(
                title = cms.untracked.string('rate')
            ),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('SuperCrystal'),
            description = cms.untracked.string('Occurrence rate of full readout when unit is flagged as zero-suppressed.')
        ),
        FRDropped = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT FR Flagged Dropped Readout%(suffix)s'),
            kind = cms.untracked.string('TH2F'),
            zaxis = cms.untracked.PSet(
                title = cms.untracked.string('rate')
            ),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('SuperCrystal'),
            description = cms.untracked.string('Occurrence rate of unit drop when the unit is flagged as full-readout.')
        )
    )
)
