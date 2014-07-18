import FWCore.ParameterSet.Config as cms

ecalPresampleTask = cms.untracked.PSet(
    params = cms.untracked.PSet(
        pulseMaxPosition = cms.untracked.int32(5),
        nSamples = cms.untracked.int32(3)
    ),
    MEs = cms.untracked.PSet(
        Pedestal = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sPedestalOnlineTask/Gain12/%(prefix)sPOT pedestal %(sm)s G12'),
            kind = cms.untracked.string('TProfile2D'),
            otype = cms.untracked.string('SM'),
            btype = cms.untracked.string('Crystal'),
            description = cms.untracked.string('2D distribution of mean presample value.')
        )
    )
)
