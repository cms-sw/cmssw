import FWCore.ParameterSet.Config as cms

ntuple_egammaEE = cms.PSet(
    NtupleName = cms.string('L1TriggerNtupleEgamma'),
    Egamma = cms.InputTag('l1EGammaEEProducer:L1EGammaCollectionBXVWithCuts'),
    BranchNamePrefix = cms.untracked.string("egammaEE")
)

ntuple_TTTracks = cms.PSet(
    NtupleName = cms.string('L1TriggerNtupleTrackTrigger'),
    TTTracks = cms.InputTag("TTTracksFromTrackletEmulation", "Level1TTTracks"),
    BranchNamePrefix = cms.untracked.string("l1Trk")
)

ntuple_tkEle = cms.PSet(
    NtupleName = cms.string('L1TriggerNtupleTkElectrons'),
    TkElectrons = cms.InputTag("L1TkElectrons","EG"),
    BranchNamePrefix = cms.untracked.string("tkEle")
)
