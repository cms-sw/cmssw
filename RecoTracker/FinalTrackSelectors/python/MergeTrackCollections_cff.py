import FWCore.ParameterSet.Config as cms

#   input1:   firstStepTracksWithQuality
#   input2:   secStep
#   output:   generalTracks
#   sequence: trackCollectionMerging
import copy
from RecoTracker.FinalTrackSelectors.ctfrsTrackListMerger_cfi import *
generalTracks = copy.deepcopy(ctfrsTrackListMerger)

trackCollectionMerging = cms.Sequence(generalTracks)

generalTracks.TrackProducer1 = 'firstStepTracksWithQuality'
generalTracks.TrackProducer2 = 'secStep'
generalTracks.promoteTrackQuality = True

