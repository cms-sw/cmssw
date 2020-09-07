import FWCore.ParameterSet.Config as cms

ntuple_egammaEE = cms.PSet(
    NtupleName = cms.string('L1TriggerNtupleEgamma'),
    Egamma = cms.InputTag('l1EGammaEEProducer:L1EGammaCollectionBXVWithCuts'),
    BranchNamePrefix = cms.untracked.string("egammaEE")
)

ntuple_egammaEB = cms.PSet(
    NtupleName = cms.string('L1TriggerNtupleEgamma'),
    Egamma = cms.InputTag("L1EGammaClusterEmuProducer"),
    BranchNamePrefix = cms.untracked.string("egammaEB")
)

ntuple_TTTracks = cms.PSet(
    NtupleName = cms.string('L1TriggerNtupleTrackTrigger'),
    TTTracks = cms.InputTag("TTTracksFromTrackletEmulation", "Level1TTTracks"),
    BranchNamePrefix = cms.untracked.string("l1Trk")
)

ntuple_tkEleEE = cms.PSet(
    NtupleName = cms.string('L1TriggerNtupleTkElectrons'),
    TkElectrons = cms.InputTag("L1TkElectronsHGC","EG"),
    BranchNamePrefix = cms.untracked.string("tkEleEE")
)

ntuple_tkEleEB = cms.PSet(
    NtupleName = cms.string('L1TriggerNtupleTkElectrons'),
    TkElectrons = cms.InputTag("L1TkElectronsCrystal","EG"),
    BranchNamePrefix = cms.untracked.string("tkEleEB")
)

ntuple_tkEleEllEE = cms.PSet(
    NtupleName = cms.string('L1TriggerNtupleTkElectrons'),
    TkElectrons = cms.InputTag("L1TkElectronsEllipticMatchHGC","EG"),
    BranchNamePrefix = cms.untracked.string("tkEleEE")
)

ntuple_tkEleEllEB = cms.PSet(
    NtupleName = cms.string('L1TriggerNtupleTkElectrons'),
    TkElectrons = cms.InputTag("L1TkElectronsEllipticMatchCrystal","EG"),
    BranchNamePrefix = cms.untracked.string("tkEleEB")
)

ntuple_tkIsoEleEE = cms.PSet(
    NtupleName = cms.string('L1TriggerNtupleTkElectrons'),
    TkElectrons = cms.InputTag("L1TkIsoElectronsHGC","EG"),
    BranchNamePrefix = cms.untracked.string("tkIsoEleEE")
)

ntuple_tkIsoEleEB = cms.PSet(
    NtupleName = cms.string('L1TriggerNtupleTkElectrons'),
    TkElectrons = cms.InputTag("L1TkIsoElectronsCrystal","EG"),
    BranchNamePrefix = cms.untracked.string("tkIsoEleEB")
)
