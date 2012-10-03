import FWCore.ParameterSet.Config as cms

ecalPnDiodeTask = cms.untracked.PSet(
    MEs = cms.untracked.PSet(
        OccupancySummary = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sOT PN digi occupancy summary'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('Crystal'),
            description = cms.untracked.string('Occupancy of PN digis in calibration events.')
        ),
        MEMTowerId = cms.untracked.PSet(
            path = cms.untracked.string('Ecal/Errors/Integrity/MEMTowerId/'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Channel'),
            btype = cms.untracked.string('Crystal'),
            description = cms.untracked.string('')
        ),
        MEMBlockSize = cms.untracked.PSet(
            path = cms.untracked.string('Ecal/Errors/Integrity/MEMBlockSize/'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Channel'),
            btype = cms.untracked.string('Crystal'),
            description = cms.untracked.string('')
        ),
        MEMChId = cms.untracked.PSet(
            path = cms.untracked.string('Ecal/Errors/Integrity/MEMChId/'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Channel'),
            btype = cms.untracked.string('Crystal'),
            description = cms.untracked.string('')
        ),
        Occupancy = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT MEM digi occupancy %(sm)s'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('SMMEM'),
            btype = cms.untracked.string('Crystal'),
            description = cms.untracked.string('Occupancy of PN digis in calibration events.')
        ),
        MEMGain = cms.untracked.PSet(
            path = cms.untracked.string('Ecal/Errors/Integrity/MEMGain/'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Channel'),
            btype = cms.untracked.string('Crystal'),
            description = cms.untracked.string('')
        ),
        Pedestal = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sPedestalOnlineTask/PN/%(prefix)sPOT PN pedestal %(sm)s G16'),
            kind = cms.untracked.string('TProfile'),
            otype = cms.untracked.string('SMMEM'),
            btype = cms.untracked.string('Crystal'),
            description = cms.untracked.string('Presample mean of PN signals.')
        )
    )
)
