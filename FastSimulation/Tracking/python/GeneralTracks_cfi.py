import FWCore.ParameterSet.Config as cms

import RecoTracker.FinalTrackSelectors.trackListMerger_cfi
generalTracksBase = RecoTracker.FinalTrackSelectors.trackListMerger_cfi.trackListMerger.clone(
    TrackProducers = (cms.InputTag('iterativeInitialTracks'),
                      cms.InputTag('iterativeDetachedTripletTracks'),
                      cms.InputTag('iterativeLowPtTripletTracksWithTriplets'),
                      cms.InputTag('iterativePixelPairTracks'),
                      cms.InputTag('iterativeMixedTripletStepTracks'),
                      cms.InputTag('iterativePixelLessTracks'),
                      cms.InputTag('iterativeTobTecTracks'),
                      ),
    hasSelector=cms.vint32(1,1,1,1,1,1,1),
    selectedTrackQuals = cms.VInputTag(cms.InputTag("initialStepSelector","initialStep"),
                                       cms.InputTag("detachedTripletStep"),
                                       cms.InputTag("lowPtTripletStepSelector","lowPtTripletStep"),
                                       cms.InputTag("pixelPairStepSelector","pixelPairStep"),
                                       cms.InputTag("mixedTripletStep"),
                                       cms.InputTag("pixelLessStepSelector","pixelLessStep"),
                                       cms.InputTag("tobTecStepSelector","tobTecStep"),
                                       ),
    setsToMerge = cms.VPSet( cms.PSet( tLists=cms.vint32(0,1,2,3,4,5,6), pQual=cms.bool(True) )                             ),
    copyExtras = True,
    makeReKeyedSeeds = cms.untracked.bool(False)
    )

generalTracksBeforeMixing = generalTracksBase.clone()
