import FWCore.ParameterSet.Config as cms

ntuple_egammaEE = cms.PSet(
    NtupleName = cms.string('L1TriggerNtupleEgammaEE'),
    EgammaEE = cms.InputTag('l1EGammaEEProducer:L1EGammaCollectionBXVWithCuts')
)

ntuple_TTTracks = cms.PSet(
    NtupleName = cms.string('L1TriggerNtupleTrackTrigger'),
    TTTracks = cms.InputTag("TTTracksFromTrackletEmulation", "Level1TTTracks")
)

ntuple_tkEle = cms.PSet(
    NtupleName = cms.string('L1TriggerNtupleTkElectrons'),
    TkElectrons = cms.InputTag("L1TkElectrons","EG")
)
