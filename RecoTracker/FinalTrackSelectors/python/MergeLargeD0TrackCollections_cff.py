import FWCore.ParameterSet.Config as cms

#
# Merge standard tracking with additional, optional large impact-parameter tracking iterations
#

import RecoTracker.FinalTrackSelectors.ctfrsTrackListMerger_cfi
# Track filtering and quality.
#   input:    generalTracks,largeD0step1WithMaterialTracks,largeD0step2WithMaterialTracks,largeD0step3WithMaterialTracks,largeD0step4WithMaterialTracks,largeD0step5WithMaterialTracks
#   output:   mergeLargeD0step5Step
#   sequence: largeD0TrackCollectionMerging

#
mergeLargeD0step1 = RecoTracker.FinalTrackSelectors.ctfrsTrackListMerger_cfi.ctfrsTrackListMerger.clone()
mergeLargeD0step1.TrackProducer1 = 'generalTracks'
mergeLargeD0step1.TrackProducer2 = 'largeD0step1Trk'
mergeLargeD0step1.promoteTrackQuality = True

#
mergeLargeD0step2 = RecoTracker.FinalTrackSelectors.ctfrsTrackListMerger_cfi.ctfrsTrackListMerger.clone()
mergeLargeD0step2.TrackProducer1 = 'mergeLargeD0step1'
mergeLargeD0step2.TrackProducer2 = 'largeD0step2Trk'
mergeLargeD0step2.promoteTrackQuality = True

#
mergeLargeD0step3 = RecoTracker.FinalTrackSelectors.ctfrsTrackListMerger_cfi.ctfrsTrackListMerger.clone()
mergeLargeD0step3.TrackProducer1 = 'mergeLargeD0step2'
mergeLargeD0step3.TrackProducer2 = 'largeD0step3Trk'
mergeLargeD0step3.promoteTrackQuality = True

#
mergeLargeD0step4 = RecoTracker.FinalTrackSelectors.ctfrsTrackListMerger_cfi.ctfrsTrackListMerger.clone()
mergeLargeD0step4.TrackProducer1 = 'mergeLargeD0step3'
mergeLargeD0step4.TrackProducer2 = 'largeD0step4Trk'
mergeLargeD0step4.promoteTrackQuality = True

#
mergeLargeD0step5 = RecoTracker.FinalTrackSelectors.ctfrsTrackListMerger_cfi.ctfrsTrackListMerger.clone()
mergeLargeD0step5.TrackProducer1 = 'mergeLargeD0step4'
mergeLargeD0step5.TrackProducer2 = 'largeD0step5Trk'
mergeLargeD0step5.promoteTrackQuality = True

largeD0TrackCollectionMerging = cms.Sequence(mergeLargeD0step1 * mergeLargeD0step2 * mergeLargeD0step3 * mergeLargeD0step4 * mergeLargeD0step5) 
