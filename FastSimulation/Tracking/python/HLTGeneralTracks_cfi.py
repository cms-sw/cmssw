import FWCore.ParameterSet.Config as cms

import RecoTracker.FinalTrackSelectors.trackListMerger_cfi
HLTgeneralTracks = RecoTracker.FinalTrackSelectors.trackListMerger_cfi.trackListMerger.clone( #temporary; substitute with just a clone!
    TrackProducers = (cms.InputTag('initialStepTracks'),
                      cms.InputTag('lowPtTripletStepTracks'),
                      cms.InputTag('pixelPairStepTracks'),
                      cms.InputTag('detachedTripletStepTracks'),
                      cms.InputTag('mixedTripletStepTracks'),
                      cms.InputTag('pixelLessStepTracks'),
                      cms.InputTag('tobTecStepTracks')),
    hasSelector=cms.vint32(1,1,1,1,1,1,1),
    selectedTrackQuals = cms.VInputTag(cms.InputTag("initialStep"),
                                       cms.InputTag("lowPtTripletStepSelector","lowPtTripletStep"),
                                       cms.InputTag("pixelPairStepSelector","pixelPairStep"),
                                       cms.InputTag("detachedTripletStep"),
                                       cms.InputTag("mixedTripletStep"),
                                       cms.InputTag("pixelLessStep"),
                                       cms.InputTag("tobTecStepSelector","tobTecStep")
                                       ),
    setsToMerge = cms.VPSet( cms.PSet( tLists=cms.vint32(0,1,2,3,4,5,6), pQual=cms.bool(True) )
                             ),
    copyExtras = True,
    makeReKeyedSeeds = cms.untracked.bool(False)
    )
