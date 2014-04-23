import FWCore.ParameterSet.Config as cms

from DQM.EcalCommon.CommonParams_cfi import *

ecalPedestalTask = cms.untracked.PSet(
    params = cms.untracked.PSet(
        MGPAGains = ecaldqmMGPAGains,
        MGPAGainsPN = ecaldqmMGPAGainsPN
    ),
    MEs = cms.untracked.PSet(
        Pedestal = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sPedestalTask/Gain%(gain)s/%(prefix)sPT pedestal %(sm)s G%(gain)s'),
            otype = cms.untracked.string('SM'),
            multi = cms.untracked.PSet(
                gain = ecaldqmMGPAGains
            ),
            kind = cms.untracked.string('TProfile2D'),
            btype = cms.untracked.string('Crystal'),
            description = cms.untracked.string('2D distribution of the mean pedestal.')
        ),
        PNPedestal = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sPedestalTask/PN/Gain%(pngain)s/%(prefix)sPDT PNs pedestal %(sm)s G%(pngain)s'),
            otype = cms.untracked.string('SMMEM'),
            multi = cms.untracked.PSet(
                pngain = ecaldqmMGPAGainsPN
            ),
            kind = cms.untracked.string('TProfile'),
            btype = cms.untracked.string('Crystal'),
            description = cms.untracked.string('Pedestal distribution of the PN diodes.')
        ),
        Occupancy = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT pedestal digi occupancy%(suffix)s'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('SuperCrystal'),
            description = cms.untracked.string('Channel occupancy in pedestal events.')
        )
    )
)
