import FWCore.ParameterSet.Config as cms

higgsToWW2LeptonsFilter = cms.EDFilter("HiggsToWW2LeptonsSkim",
    ElectronCollectionLabel = cms.InputTag("pixelMatchGsfElectrons"),
    RecoTrackLabel = cms.InputTag("recoTracks"),
    SingleTrackPtMin = cms.double(20.0),
    etaMin = cms.double(-2.4),
    GlobalMuonCollectionLabel = cms.InputTag("globalMuons"),
    DiTrackPtMin = cms.double(10.0),
    etaMax = cms.double(2.4)
)


