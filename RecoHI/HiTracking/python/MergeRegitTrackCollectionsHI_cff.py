import FWCore.ParameterSet.Config as cms

from RecoTracker.FinalTrackSelectors.trackAlgoPriorityOrder_cfi import trackAlgoPriorityOrder
import RecoTracker.FinalTrackSelectors.trackListMerger_cfi
hiRegitTracks = RecoTracker.FinalTrackSelectors.trackListMerger_cfi.trackListMerger.clone(
    TrackProducers = ['hiRegitInitialStepTracks',
                      'hiRegitLowPtTripletStepTracks',
                      'hiRegitPixelPairStepTracks',
                      'hiRegitDetachedTripletStepTracks',
                      'hiRegitMixedTripletStepTracks'
		     ],
    hasSelector = [1,1,1,1,1],
    selectedTrackQuals = ["hiRegitInitialStepSelector:hiRegitInitialStep",
			  "hiRegitLowPtTripletStepSelector:hiRegitLowPtTripletStep",
			  "hiRegitPixelPairStepSelector:hiRegitPixelPairStep",
			  "hiRegitDetachedTripletStepSelector:hiRegitDetachedTripletStep",
			  "hiRegitMixedTripletStepSelector:hiRegitMixedTripletStep"
			 ],                    
    setsToMerge = cms.VPSet( cms.PSet( tLists=cms.vint32(0,1,2,3,4,5), pQual=cms.bool(True)),  # should this be False?
                             ),
    copyExtras = True,
    makeReKeyedSeeds = cms.untracked.bool(False)
    )
