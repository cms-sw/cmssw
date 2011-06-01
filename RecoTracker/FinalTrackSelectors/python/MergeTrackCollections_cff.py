import FWCore.ParameterSet.Config as cms

import RecoTracker.FinalTrackSelectors.simpleTrackListMerger_cfi
firstStepTracksWithQuality = RecoTracker.FinalTrackSelectors.simpleTrackListMerger_cfi.simpleTrackListMerger.clone(
    TrackProducer1 = 'zeroStepTracksWithQuality',
    TrackProducer2 = 'preMergingFirstStepTracksWithQuality',
    promoteTrackQuality = False
    )


# new merging module
import RecoTracker.FinalTrackSelectors.trackListMerger_cfi
generalTracks = RecoTracker.FinalTrackSelectors.trackListMerger_cfi.trackListMerger.clone(
    TrackProducers = ('firstStepTracksWithQuality',
                      'secWithMaterialTracks',
                      'thWithMaterialTracks',
                      'fourthWithMaterialTracks',
                      'fifthWithMaterialTracks'),
    hasSelector=cms.vint32(0,1,1,1,1),
    selectedTrackQuals = cms.VInputTag(cms.InputTag(""),
                                       cms.InputTag("secStep"),
                                       cms.InputTag("thStep"),
                                       cms.InputTag("pixellessSelector","pixellessStep"),
                                       cms.InputTag("tobtecSelector","tobtecStep")
                                       ),
    setsToMerge = cms.VPSet( cms.PSet( tLists=cms.vint32(1,2), pQual=cms.bool(True) ),
                             cms.PSet( tLists=cms.vint32(3,4), pQual=cms.bool(True) ),
                             cms.PSet( tLists=cms.vint32(1,2,3,4), pQual=cms.bool(True) ),
                             cms.PSet( tLists=cms.vint32(0,1,2,3,4), pQual=cms.bool(True) )
                             ),
    copyExtras = True,
    makeReKeyedSeeds = cms.untracked.bool(True)
    )

trackCollectionMerging = cms.Sequence(generalTracks)

