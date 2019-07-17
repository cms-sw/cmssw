import FWCore.ParameterSet.Config as cms

l1TrkObjAnaysis = cms.EDAnalyzer( 'L1TkObjectAnalyzer' ,
    L1EGammaInputTag = cms.InputTag("simCaloStage2Digis",""),
    L1MuonInputTag   = cms.InputTag("simGmtStage2Digis",""),
    L1TkMuonInputTag     = cms.InputTag("L1TkMuons",""),
    L1TkPhotonInputTag   = cms.InputTag("L1TkPhotons", "EG"),
    L1TkElectronInputTag = cms.InputTag("L1TkElectrons","EG"),
    L1TrackInputTag = cms.InputTag("TTTracksFromTracklet", "Level1TTTracks"),
    GenParticleInputTag = cms.InputTag("genParticles",""),
    AnalysisOption   = cms.string("Efficiency"),
    ObjectType       = cms.string("Electron"),
    EtaCutOff   = cms.double(2.5),
    TrackPtCutOff   = cms.double(10.0),
    GenPtThreshold   = cms.double(20.0),
    EtThreshold = cms.double(20.0)                              
)
