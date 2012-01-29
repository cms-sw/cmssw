import FWCore.ParameterSet.Config as cms

import RecoTracker.FinalTrackSelectors.trackListMerger_cfi
hiGeneralTracks = RecoTracker.FinalTrackSelectors.trackListMerger_cfi.trackListMerger.clone(
    TrackProducers = (cms.InputTag('hiGlobalPrimTracks'),
                      cms.InputTag('hiSecondPixelTripletGlobalPrimTracks'),
                      cms.InputTag('hiMixedTripletGlobalPrimTracks'),
                      #cms.InputTag('detachedTripletStepTracks'),
                      cms.InputTag('hiPixelPairGlobalPrimTracks')),
    #cms.InputTag('pixelLessStepTracks'),
    #cms.InputTag('tobTecStepTracks')),
    hasSelector=cms.vint32(1,1,1,1),
    selectedTrackQuals = cms.VInputTag(
    cms.InputTag("hiInitialStepSelector","hiInitialStep"),
    cms.InputTag("hiSecondPixelTripletStepSelector","hiSecondPixelTripletStep"),
    cms.InputTag("hiMixedTripletStepSelector","hiMixedTripletStep"),
    cms.InputTag("hiPixelPairStepSelector","hiPixelPairStep"),
    ),                    
    setsToMerge = cms.VPSet( cms.PSet( tLists=cms.vint32(0,1,2,3), pQual=cms.bool(True)),  # should this be False?
                             ),
    copyExtras = True,
    makeReKeyedSeeds = cms.untracked.bool(False)
    )
