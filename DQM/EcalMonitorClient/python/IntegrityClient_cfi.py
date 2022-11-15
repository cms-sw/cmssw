import FWCore.ParameterSet.Config as cms

from DQM.EcalMonitorTasks.OccupancyTask_cfi import ecalOccupancyTask
from DQM.EcalMonitorTasks.IntegrityTask_cfi import ecalIntegrityTask
from DQM.EcalMonitorTasks.RawDataTask_cfi import ecalRawDataTask

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
        BXSRP = ecalRawDataTask.MEs.BXSRP,
        BXTCC = ecalRawDataTask.MEs.BXTCC,
	NumEvents = ecalOccupancyTask.MEs.NEvents
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
            description = cms.untracked.string('Map of channel status as given by the Ecal Channel Status Record. LEGEND:<br/>0: Channel ok,<br/>1: DAC settings problem, pedestal not in the design range,<br/>2: Channel with no laser, ok elsewhere,<br/>3: Noisy,<br/>4: Very noisy,<br/>5-7: Reserved for more categories of noisy channels,<br/>8: Channel at fixed gain 6 (or 6 and 1),<br/>9: Channel at fixed gain 1,<br/>10: Channel at fixed gain 0 (dead of type this),<br/>11: Non-responding isolated channel (dead of type other),<br/>12: Channel and one or more neigbors not responding (e.g.: in a dead VFE 5x1 channel),<br/>13: Channel in TT with no data link, TP data ok,<br/>14: Channel in TT with no data link and no TP data.')
       ),
	TowerIdNormalized = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sIntegrityTask/TTId/%(prefix)sIT TTId Normalized %(sm)s'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('SM'),
            btype = cms.untracked.string('SuperCrystal'),
            description = cms.untracked.string('TTID errors normalized by total no.of processed events per run.')
       )
    )
)
