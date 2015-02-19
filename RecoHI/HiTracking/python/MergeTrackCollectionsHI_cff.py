import FWCore.ParameterSet.Config as cms

import RecoTracker.FinalTrackSelectors.trackListMerger_cfi
hiGeneralTracksNoRegitMu = RecoTracker.FinalTrackSelectors.trackListMerger_cfi.trackListMerger.clone(
    TrackProducers = (cms.InputTag('hiGlobalPrimTracks'),
                      cms.InputTag('hiDetachedTripletStepTracks'),
                      cms.InputTag('hiLowPtTripletStepTracks'),
                      cms.InputTag('hiPixelPairGlobalPrimTracks')
                     ),
    hasSelector=cms.vint32(1,1,1,1),
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
                      cms.InputTag('hiRegitMuInitialStepTracks'),
                      cms.InputTag('hiRegitMuLowPtTripletStepTracks'),
                      cms.InputTag('hiRegitMuPixelPairStepTracks'),
                      cms.InputTag('hiRegitMuDetachedTripletStepTracks'),
                      cms.InputTag('hiRegitMuMixedTripletStepTracks'),
                      cms.InputTag('hiRegitMuPixelLessStepTracks'),
                      cms.InputTag('hiRegitMuTobTecStepTracks')
                     ),
    hasSelector=cms.vint32(1,1,1,1,1,1,1,1,1,1,1),
    selectedTrackQuals = cms.VInputTag(
    cms.InputTag("hiInitialStepSelector","hiInitialStep"),
    cms.InputTag("hiDetachedTripletStepSelector","hiDetachedTripletStep"),
    cms.InputTag("hiLowPtTripletStepSelector","hiLowPtTripletStep"),
    cms.InputTag("hiPixelPairStepSelector","hiPixelPairStep"),
    cms.InputTag("hiRegitMuInitialStepSelector","hiRegitMuInitialStepLoose"),
    cms.InputTag("hiRegitMuLowPtTripletStepSelector","hiRegitMuLowPtTripletStepLoose"),
    cms.InputTag("hiRegitMuPixelPairStepSelector","hiRegitMuPixelPairStep"),
    cms.InputTag("hiRegitMuDetachedTripletStepSelector","hiRegitMuDetachedTripletStep"),
    cms.InputTag("hiRegitMuMixedTripletStepSelector","hiRegitMuMixedTripletStep"),
    cms.InputTag("hiRegitMuPixelLessStepSelector","hiRegitMuPixelLessStep"),
    cms.InputTag("hiRegitMuTobTecStepSelector","hiRegitMuTobTecStep")
    ),                    
    setsToMerge = cms.VPSet( cms.PSet( tLists=cms.vint32(0,1,2,3,4,5,6,7,8,9,10), pQual=cms.bool(True)),  # should this be False?
                             ),
    copyExtras = True,
    makeReKeyedSeeds = cms.untracked.bool(False)
    )
