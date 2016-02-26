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
        ),
        ChStatus = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sIntegrityClient/%(prefix)sIT%(suffix)s channel status map'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('Crystal'),
            description = cms.untracked.string('Map of channel status as given by the Ecal Channel Status Record. LEGEND: 0:channel ok, 1:DAC settings problem, pedestal not in the design range, 2:channel with no laser, ok elsewhere, 3:noisy, 4:very noisy, 5-7:reserved for more categories of noisy channels, 8:channel at fixed gain 6 (or 6 and 1), 9:channel at fixed gain 1, 10:channel at fixed gain 0 (dead of type this), 11:non responding isolated channel (dead of type other), 12:channel and one or more neigbors not responding (e.g.: in a dead VFE 5x1 channel), 13:channel in TT with no data link, TP data ok, 14:channel in TT with no data link and no TP data.')
        )
    )
)
