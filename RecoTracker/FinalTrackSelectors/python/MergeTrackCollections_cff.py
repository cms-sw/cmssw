import FWCore.ParameterSet.Config as cms


import RecoTracker.FinalTrackSelectors.simpleTrackListMerger_cfi
firstStepTracksWithQuality = RecoTracker.FinalTrackSelectors.simpleTrackListMerger_cfi.simpleTrackListMerger.clone(
    TrackProducer1 = 'zeroStepTracksWithQuality',
    TrackProducer2 = 'preMergingFirstStepTracksWithQuality',
    promoteTrackQuality = False
    )


#switch this to 0 to get the new merging module
if (1): 
    # Track filtering and quality.
    #   input:    zeroStepTracksWithQuality,preMergingFirstStepTracksWithQuality,secStep,thStep,pixellessStep
    #   output:   generalTracks
    #   sequence: trackCollectionMerging
    
    #
    
    
    merge2nd3rdTracks = RecoTracker.FinalTrackSelectors.simpleTrackListMerger_cfi.simpleTrackListMerger.clone(
    TrackProducer1 = 'secStep',
    TrackProducer2 = 'thStep',
    promoteTrackQuality = True
    )
    
    merge4th5thTracks = RecoTracker.FinalTrackSelectors.simpleTrackListMerger_cfi.simpleTrackListMerger.clone(
    TrackProducer1 = 'pixellessStep',
    TrackProducer2 = 'tobtecStep',
    promoteTrackQuality = True
    )
    
    iterTracks = RecoTracker.FinalTrackSelectors.simpleTrackListMerger_cfi.simpleTrackListMerger.clone(
    TrackProducer1 = 'merge2nd3rdTracks',
    TrackProducer2 = 'merge4th5thTracks',
    promoteTrackQuality = True
    )
    
    generalTracks = RecoTracker.FinalTrackSelectors.simpleTrackListMerger_cfi.simpleTrackListMerger.clone(
    TrackProducer1 = 'firstStepTracksWithQuality',
    TrackProducer2 = 'iterTracks',
    promoteTrackQuality = True,
    copyExtras = True,
    makeReKeyedSeeds = cms.untracked.bool(True)
    )
    
    trackCollectionMerging = cms.Sequence(merge2nd3rdTracks*
                                          merge4th5thTracks*
                                          iterTracks*
                                          generalTracks)

    

else:
# new merging module
    import RecoTracker.FinalTrackSelectors.trackListMerger_cfi
    generalTracks = RecoTracker.FinalTrackSelectors.trackListMerger_cfi.trackListMerger.clone(
    TrackProducers = ('firstStepTracksWithQuality','secStep','thStep','pixellessStep','tobtecStep'),
    setsToMerge = cms.VPSet( cms.PSet( tLists=cms.vint32(1,2), pQual=cms.bool(True) ),
                             cms.PSet( tLists=cms.vint32(3,4), pQual=cms.bool(True) ),
                             cms.PSet( tLists=cms.vint32(1,2,3,4), pQual=cms.bool(True) ),
                             cms.PSet( tLists=cms.vint32(0,1,2,3,4), pQual=cms.bool(True) )
                             ),

#this could work if the firstStepTracksWithQuality module above could be removed
#    TrackProducers = ('zeroStepTracksWithQuality','preMergingFirstStepTracksWithQuality','secStep','thStep','pixellessStep','tobtecStep'),
#        setsToMerge = cms.VPSet( cms.PSet( tLists=cms.vint32(0,1), pQual=cms.bool(False)),
#                             cms.PSet( tLists=cms.vint32(2,3), pQual=cms.bool(True) ),
#                             cms.PSet( tLists=cms.vint32(4,5), pQual=cms.bool(True) ),
#                             cms.PSet( tLists=cms.vint32(2,3,4,5), pQual=cms.bool(True) ),
#                             cms.PSet( tLists=cms.vint32(0,1,2,3,4,5), pQual=cms.bool(True) )
#                             ),

    copyExtras = True,
    makeReKeyedSeeds = cms.untracked.bool(True)
    )
    
    trackCollectionMerging = cms.Sequence(generalTracks)

