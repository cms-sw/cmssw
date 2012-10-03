import FWCore.ParameterSet.Config as cms

from DQM.EcalBarrelMonitorTasks.PNDiodeTask_cfi import ecalPnDiodeTask

errFractionThreshold = 0.01

ecalPnIntegrityClient = cms.untracked.PSet(
    errFractionThreshold = cms.untracked.double(errFractionThreshold),
    sources = cms.untracked.PSet(
        Occupancy = ecalPnDiodeTask.MEs.Occupancy,
        MEMChId = ecalPnDiodeTask.MEs.MEMChId,
        MEMGain = ecalPnDiodeTask.MEs.MEMGain,
        MEMBlockSize = ecalPnDiodeTask.MEs.MEMBlockSize,
        MEMTowerId = ecalPnDiodeTask.MEs.MEMTowerId
    ),
    MEs = cms.untracked.PSet(
        QualitySummary = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sIT PN integrity quality summary'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('MEM2P'),
            btype = cms.untracked.string('Crystal'),
            description = cms.untracked.string('Summary of the data integrity in PN channels. A channel is red if more than ' + str(100 * errFractionThreshold) + '% of its entries have integrity errors.')
        )
    )
)
