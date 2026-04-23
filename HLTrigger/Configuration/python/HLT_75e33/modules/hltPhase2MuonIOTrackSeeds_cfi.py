import FWCore.ParameterSet.Config as cms

hltPhase2MuonIOTrackSeeds = cms.EDProducer('MuonSeedsSelectorFromL1TkMuon',
    L1TkMuonInputCollection = cms.InputTag('l1tTkMuonsGmt'),
    SeedInputCollection = cms.InputTag('hltInitialStepTrajectorySeedsLST'),
    beamSpot = cms.InputTag('hltOnlineBeamSpot'),
    L1TkMuMinPt = cms.double(0),
    maxDrForPreselection = cms.double(0.4),
    pTCompatibility = cms.double(0.5),
    maxPtForCompatibilityCheck = cms.double(25),
    maxCombinedQuality = cms.double(2),
    doConsistencyCheckWithL1Stubs = cms.bool(False),
    minNPixelHits = cms.int32(2),
    minNConsistentHits = cms.int32(4),
    nSeedsToKeep = cms.uint32(2)
)
