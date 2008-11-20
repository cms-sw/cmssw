import FWCore.ParameterSet.Config as cms

import RecoTracker.FinalTrackSelectors.ctfrsTrackListMerger_cfi
# Track filtering and quality.
#   input:    zeroStepTracksWithQuality,preMergingFirstStepTracksWithQuality,secStep,thStep,pixellessStep
#   output:   generalTracks
#   sequence: trackCollectionMerging

#

firstStepTracksWithQuality = RecoTracker.FinalTrackSelectors.ctfrsTrackListMerger_cfi.ctfrsTrackListMerger.clone()
firstStepTracksWithQuality.TrackProducer1 = 'zeroStepTracksWithQuality'
firstStepTracksWithQuality.TrackProducer2 = 'preMergingFirstStepTracksWithQuality'
firstStepTracksWithQuality.promoteTrackQuality = False


merge2nd3rdTracks = RecoTracker.FinalTrackSelectors.ctfrsTrackListMerger_cfi.ctfrsTrackListMerger.clone()
merge2nd3rdTracks.TrackProducer1 = 'secStep'
merge2nd3rdTracks.TrackProducer2 = 'thStep'
merge2nd3rdTracks.promoteTrackQuality = True

merge4th5thTracks = RecoTracker.FinalTrackSelectors.ctfrsTrackListMerger_cfi.ctfrsTrackListMerger.clone()
merge4th5thTracks.TrackProducer1 = 'pixellessStep'
merge4th5thTracks.TrackProducer2 = 'tobtecStep'
merge4th5thTracks.promoteTrackQuality = True

iterTracks = RecoTracker.FinalTrackSelectors.ctfrsTrackListMerger_cfi.ctfrsTrackListMerger.clone()
iterTracks.TrackProducer1 = 'merge2nd3rdTracks'
iterTracks.TrackProducer2 = 'merge4th5thTracks'
iterTracks.promoteTrackQuality = True

generalTracks = RecoTracker.FinalTrackSelectors.ctfrsTrackListMerger_cfi.ctfrsTrackListMerger.clone()
generalTracks.TrackProducer1 = 'firstStepTracksWithQuality'
generalTracks.TrackProducer2 = 'iterTracks'
generalTracks.promoteTrackQuality = True

trackCollectionMerging = cms.Sequence(merge2nd3rdTracks*
                                      merge4th5thTracks*
                                      iterTracks*
                                      generalTracks)
