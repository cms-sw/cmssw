import FWCore.ParameterSet.Config as cms

import RecoTracker.FinalTrackSelectors.trackListMerger_cfi
hiGeneralTracksNoRegitMu = RecoTracker.FinalTrackSelectors.trackListMerger_cfi.trackListMerger.clone(
    TrackProducers = (cms.InputTag('hiGlobalPrimTracks'),
                      cms.InputTag('hiDetachedTripletStepTracks'),
                      cms.InputTag('hiLowPtTripletStepTracks'),
                      cms.InputTag('hiPixelPairGlobalPrimTracks'),
                      cms.InputTag('hiJetCoreRegionalStepTracks')
                     ),
    hasSelector=cms.vint32(1,1,1,1,1),
    selectedTrackQuals = cms.VInputTag(
    cms.InputTag("hiInitialStepSelector","hiInitialStep"),
    cms.InputTag("hiDetachedTripletStepSelector","hiDetachedTripletStep"),
    cms.InputTag("hiLowPtTripletStepSelector","hiLowPtTripletStep"),
    cms.InputTag("hiPixelPairStepSelector","hiPixelPairStep"),
    ),                    
    setsToMerge = cms.VPSet( cms.PSet( tLists=cms.vint32(0,1,2,3), pQual=cms.bool(True)),  # should this be False?
                             ),
    copyExtras = True,
    makeReKeyedSeeds = cms.untracked.bool(False)
    )

hiGeneralTracks = RecoTracker.FinalTrackSelectors.trackListMerger_cfi.trackListMerger.clone(
    TrackProducers = (cms.InputTag('hiGlobalPrimTracks'),
                      cms.InputTag('hiDetachedTripletStepTracks'),
                      cms.InputTag('hiLowPtTripletStepTracks'),
                      cms.InputTag('hiPixelPairGlobalPrimTracks'),
                      cms.InputTag('hiJetCoreRegionalStepTracks'),
                      cms.InputTag('hiRegitMuInitialStepTracks'),
                      cms.InputTag('hiRegitMuPixelPairStepTracks'),
                      cms.InputTag('hiRegitMuMixedTripletStepTracks'),
                      cms.InputTag('hiRegitMuPixelLessStepTracks'),
                      cms.InputTag('hiRegitMuDetachedTripletStepTracks'),
                      cms.InputTag('hiRegitMuonSeededTracksOutIn'),
                      cms.InputTag('hiRegitMuonSeededTracksInOut')
                     ),
    hasSelector=cms.vint32(1,1,1,1,1,1,1,1,1,1,1,1),
    selectedTrackQuals = cms.VInputTag(
    cms.InputTag("hiInitialStepSelector","hiInitialStep"),
    cms.InputTag("hiDetachedTripletStepSelector","hiDetachedTripletStep"),
    cms.InputTag("hiLowPtTripletStepSelector","hiLowPtTripletStep"),
    cms.InputTag("hiPixelPairStepSelector","hiPixelPairStep"),
    cms.InputTag("hiJetCoreRegionalStepSelector","hiJetCoreRegionalStep"),
    cms.InputTag("hiRegitMuInitialStepSelector","hiRegitMuInitialStepLoose"),
    cms.InputTag("hiRegitMuPixelPairStepSelector","hiRegitMuPixelPairStep"),
    cms.InputTag("hiRegitMuMixedTripletStepSelector","hiRegitMuMixedTripletStep"),
    cms.InputTag("hiRegitMuPixelLessStepSelector","hiRegitMuPixelLessStep"),
    cms.InputTag("hiRegitMuDetachedTripletStepSelector","hiRegitMuDetachedTripletStep"),
    cms.InputTag("hiRegitMuonSeededTracksOutInSelector","hiRegitMuonSeededTracksOutInHighPurity"),
    cms.InputTag("hiRegitMuonSeededTracksInOutSelector","hiRegitMuonSeededTracksInOutHighPurity")
    ),                    
    setsToMerge = cms.VPSet( cms.PSet( tLists=cms.vint32(0,1,2,3,4,5,6,7,8,9,10,11), pQual=cms.bool(True)),  # should this be False?
                             ),
    copyExtras = True,
    makeReKeyedSeeds = cms.untracked.bool(False)
    )
