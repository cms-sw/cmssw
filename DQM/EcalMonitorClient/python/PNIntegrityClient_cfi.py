import FWCore.ParameterSet.Config as cms

from DQM.EcalMonitorTasks.PNDiodeTask_cfi import ecalPNDiodeTask

errFractionThreshold = 0.01

ecalPNIntegrityClient = cms.untracked.PSet(
    params = cms.untracked.PSet(
        errFractionThreshold = cms.untracked.double(errFractionThreshold)
    ),
    sources = cms.untracked.PSet(
        Occupancy = ecalPNDiodeTask.MEs.Occupancy,
        MEMChId = ecalPNDiodeTask.MEs.MEMChId,
        MEMGain = ecalPNDiodeTask.MEs.MEMGain,
        MEMBlockSize = ecalPNDiodeTask.MEs.MEMBlockSize,
        MEMTowerId = ecalPNDiodeTask.MEs.MEMTowerId
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
