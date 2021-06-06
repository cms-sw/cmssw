import FWCore.ParameterSet.Config as cms

import RecoTracker.FinalTrackSelectors.trackListMerger_cfi
hiGeneralTracksNoRegitMu = RecoTracker.FinalTrackSelectors.trackListMerger_cfi.trackListMerger.clone(
    TrackProducers = ['hiGlobalPrimTracks',
                      'hiDetachedTripletStepTracks',
                      'hiLowPtTripletStepTracks',
                      'hiPixelPairGlobalPrimTracks',
                      'hiJetCoreRegionalStepTracks'
                     ],
    hasSelector = [1,1,1,1,1],
    selectedTrackQuals = ["hiInitialStepSelector:hiInitialStep",
    			  "hiDetachedTripletStepSelector:hiDetachedTripletStep",
    			  "hiLowPtTripletStepSelector:hiLowPtTripletStep",
    			  "hiPixelPairStepSelector:hiPixelPairStep"
    			  ],                    
    setsToMerge = cms.VPSet( cms.PSet( tLists=cms.vint32(0,1,2,3), pQual=cms.bool(True)),  # should this be False?
                             ),
    copyExtras = True,
    makeReKeyedSeeds = cms.untracked.bool(False)
    )
from Configuration.Eras.Modifier_trackingPhase1_cff import trackingPhase1
trackingPhase1.toModify(hiGeneralTracksNoRegitMu,
    TrackProducers = ['hiGlobalPrimTracks',
                      'hiLowPtQuadStepTracks',
                      'hiHighPtTripletStepTracks',
                      'hiDetachedQuadStepTracks',
                      'hiDetachedTripletStepTracks',
                      'hiLowPtTripletStepTracks',
                      'hiPixelPairGlobalPrimTracks',
                      'hiJetCoreRegionalStepTracks'
                     ],
    hasSelector = [1,1,1,1,1,1,1,1],
    setsToMerge = cms.VPSet( cms.PSet( tLists=cms.vint32(0,1,2,3,4,5,6), pQual=cms.bool(True))),
    selectedTrackQuals = ["hiInitialStepSelector:hiInitialStep",
    			  "hiLowPtQuadStepSelector:hiLowPtQuadStep",
    			  "hiHighPtTripletStepSelector:hiHighPtTripletStep",
    			  "hiDetachedQuadStepSelector:hiDetachedQuadStep",
    			  "hiDetachedTripletStepSelector:hiDetachedTripletStep",
    			  "hiLowPtTripletStepSelector:hiLowPtTripletStep",
    			  "hiPixelPairStepSelector:hiPixelPairStep"
    			 ], 
)

hiGeneralTracks = RecoTracker.FinalTrackSelectors.trackListMerger_cfi.trackListMerger.clone(
    TrackProducers = ['hiGlobalPrimTracks',
                      'hiDetachedTripletStepTracks',
                      'hiLowPtTripletStepTracks',
                      'hiPixelPairGlobalPrimTracks',
                      'hiJetCoreRegionalStepTracks',
                      'hiRegitMuInitialStepTracks',
                      'hiRegitMuPixelPairStepTracks',
                      'hiRegitMuMixedTripletStepTracks',
                      'hiRegitMuPixelLessStepTracks',
                      'hiRegitMuDetachedTripletStepTracks',
                      'hiRegitMuonSeededTracksOutIn',
                      'hiRegitMuonSeededTracksInOut'
                     ],
    hasSelector = [1,1,1,1,1,1,1,1,1,1,1,1],
    selectedTrackQuals = ["hiInitialStepSelector:hiInitialStep",
    			  "hiDetachedTripletStepSelector:hiDetachedTripletStep",
    			  "hiLowPtTripletStepSelector:hiLowPtTripletStep",
    			  "hiPixelPairStepSelector:hiPixelPairStep",
    			  "hiJetCoreRegionalStepSelector:hiJetCoreRegionalStep",
    			  "hiRegitMuInitialStepSelector:hiRegitMuInitialStepLoose",
    			  "hiRegitMuPixelPairStepSelector:hiRegitMuPixelPairStep",
    			  "hiRegitMuMixedTripletStepSelector:hiRegitMuMixedTripletStep",
    			  "hiRegitMuPixelLessStepSelector:hiRegitMuPixelLessStep",
    			  "hiRegitMuDetachedTripletStepSelector:hiRegitMuDetachedTripletStep",
    			  "hiRegitMuonSeededTracksOutInSelector:hiRegitMuonSeededTracksOutInHighPurity",
    			  "hiRegitMuonSeededTracksInOutSelector:hiRegitMuonSeededTracksInOutHighPurity"
    			 ], 
    setsToMerge = cms.VPSet( cms.PSet( tLists=cms.vint32(0,1,2,3,4,5,6,7,8,9,10,11), pQual=cms.bool(True)),  # should this be False?
                             ),
    copyExtras = True,
    makeReKeyedSeeds = cms.untracked.bool(False)
)
trackingPhase1.toModify(hiGeneralTracks,
    TrackProducers = ['hiGlobalPrimTracks',
                      'hiLowPtQuadStepTracks',
                      'hiHighPtTripletStepTracks',
                      'hiDetachedQuadStepTracks',
                      'hiDetachedTripletStepTracks',
                      'hiLowPtTripletStepTracks',
                      'hiPixelPairGlobalPrimTracks',
                      'hiMixedTripletStepTracks',
                      'hiPixelLessStepTracks',
                      'hiTobTecStepTracks',
                      'hiJetCoreRegionalStepTracks',
                      'hiRegitMuInitialStepTracks',
                      'hiRegitMuPixelPairStepTracks',
                      'hiRegitMuMixedTripletStepTracks',
                      'hiRegitMuPixelLessStepTracks',
                      'hiRegitMuDetachedTripletStepTracks',
                      'hiRegitMuonSeededTracksOutIn',
                      'hiRegitMuonSeededTracksInOut'
                     ],
    hasSelector = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    setsToMerge = cms.VPSet( cms.PSet( tLists=cms.vint32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17), pQual=cms.bool(True))),  # should this be False?
    selectedTrackQuals = ["hiInitialStepSelector:hiInitialStep",
    			  "hiLowPtQuadStepSelector:hiLowPtQuadStep",
    			  "hiHighPtTripletStepSelector:hiHighPtTripletStep",
    			  "hiDetachedQuadStepSelector:hiDetachedQuadStep",
    			  "hiDetachedTripletStepSelector:hiDetachedTripletStep",
    			  "hiLowPtTripletStepSelector:hiLowPtTripletStep",
    			  "hiPixelPairStepSelector:hiPixelPairStep",
    			  "hiMixedTripletStepSelector:hiMixedTripletStep",
    			  "hiPixelLessStepSelector:hiPixelLessStep",
    			  "hiTobTecStepSelector:hiTobTecStep",
    			  "hiJetCoreRegionalStepSelector:hiJetCoreRegionalStep",
    			  "hiRegitMuInitialStepSelector:hiRegitMuInitialStepLoose",
    			  "hiRegitMuPixelPairStepSelector:hiRegitMuPixelPairStep",
    			  "hiRegitMuMixedTripletStepSelector:hiRegitMuMixedTripletStep",
    			  "hiRegitMuPixelLessStepSelector:hiRegitMuPixelLessStep",
    			  "hiRegitMuDetachedTripletStepSelector:hiRegitMuDetachedTripletStep",
    			  "hiRegitMuonSeededTracksOutInSelector:hiRegitMuonSeededTracksOutInHighPurity",
    			  "hiRegitMuonSeededTracksInOutSelector:hiRegitMuonSeededTracksInOutHighPurity"
    			 ],
)
