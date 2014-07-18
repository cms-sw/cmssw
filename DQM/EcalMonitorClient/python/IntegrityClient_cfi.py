import FWCore.ParameterSet.Config as cms

from DQM.EcalMonitorTasks.OccupancyTask_cfi import ecalOccupancyTask
from DQM.EcalMonitorTasks.IntegrityTask_cfi import ecalIntegrityTask

errFractionThreshold = 0.01

ecalIntegrityClient = cms.untracked.PSet(
    params = cms.untracked.PSet(
        errFractionThreshold = cms.untracked.double(errFractionThreshold)
    ),
    sources = cms.untracked.PSet(
        Occupancy = ecalOccupancyTask.MEs.Digi,
        BlockSize = ecalIntegrityTask.MEs.BlockSize,
        Gain = ecalIntegrityTask.MEs.Gain,
        GainSwitch = ecalIntegrityTask.MEs.GainSwitch,
        ChId = ecalIntegrityTask.MEs.ChId,
        TowerId = ecalIntegrityTask.MEs.TowerId,
    ),
    MEs = cms.untracked.PSet(
        QualitySummary = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sIT%(suffix)s integrity quality summary'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('Crystal'),
            description = cms.untracked.string('Summary of the data integrity. A channel is red if more than ' + str(errFractionThreshold) + ' of its entries have integrity errors.')
        ),
        Quality = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sIntegrityClient/%(prefix)sIT data integrity quality %(sm)s'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('SM'),
            btype = cms.untracked.string('Crystal'),
            description = cms.untracked.string('Summary of the data integrity. A channel is red if more than ' + str(errFractionThreshold) + ' of its entries have integrity errors.')            
        )
    )
)
