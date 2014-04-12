import FWCore.ParameterSet.Config as cms

from DQM.EcalBarrelMonitorTasks.OccupancyTask_cfi import ecalOccupancyTask

minHits = 20
deviationThreshold = 100.

ecalOccupancyClient = cms.untracked.PSet(
    params = cms.untracked.PSet(
        minHits = cms.untracked.int32(minHits),
        deviationThreshold = cms.untracked.double(deviationThreshold)
    ),
    sources = cms.untracked.PSet(
        TPDigiThrAll = ecalOccupancyTask.MEs.TPDigiThrAll,
        RecHitThrAll = ecalOccupancyTask.MEs.RecHitThrAll,
        DigiAll = ecalOccupancyTask.MEs.DigiAll
    ),
    MEs = cms.untracked.PSet(
        QualitySummary = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sOT%(suffix)s hot cell quality summary'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('SuperCrystal'),
            description = cms.untracked.string('Summary of the hot cell monitor. A channel is red if it has more than ' + str(deviationThreshold) + ' times more entries than phi-ring mean in either digi, rec hit (filtered), or TP digi (filtered). Channels with less than ' + str(minHits) + ' entries are not considered. Channel names of the hot cells are available in (Top) / Ecal / Errors / HotCells.')
        )
#        HotTPDigiThr = cms.untracked.PSet(
#            path = cms.untracked.string('Ecal/Errors/HotCells/TPDigiThres/'),
#            kind = cms.untracked.string('TH1F'),
#            otype = cms.untracked.string('Channel'),
#            btype = cms.untracked.string('TriggerTower'),
#            description = cms.untracked.string('')
#        ),
#        HotRecHitThr = cms.untracked.PSet(
#            path = cms.untracked.string('Ecal/Errors/HotCells/RecHitThres/'),
#            kind = cms.untracked.string('TH1F'),
#            otype = cms.untracked.string('Channel'),
#            btype = cms.untracked.string('Crystal'),
#            description = cms.untracked.string('')            
#        ),
#        HotDigi = cms.untracked.PSet(
#            path = cms.untracked.string('Ecal/Errors/HotCells/Digi/'),
#            kind = cms.untracked.string('TH1F'),
#            otype = cms.untracked.string('Channel'),
#            btype = cms.untracked.string('Crystal'),
#            description = cms.untracked.string('')
#        )
    )
)
