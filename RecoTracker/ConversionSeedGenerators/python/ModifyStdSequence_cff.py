import FWCore.ParameterSet.Config as cms

from RecoTracker.ConversionSeedGenerators.ConversionSequences_cff        import *
from Configuration.StandardSequences.Reconstruction_cff                  import *

generalTracksStd = generalTracks.clone()

trackCollectionMerging.remove(generalTracks)
trackCollectionMerging += generalTracksStd

mergeConversionTracks = RecoTracker.FinalTrackSelectors.simpleTrackListMerger_cfi.simpleTrackListMerger.clone(
    TrackProducer1 = 'sixthStep',
    TrackProducer2 = 'seventhStep',
    promoteTrackQuality = True
    )


generalTracks.TrackProducer2 = 'mergeConversionTracks'
generalTracks.TrackProducer1 = 'generalTracksStd'

trackCollectionMerging *= conversionStep
trackCollectionMerging *= mergeConversionTracks
trackCollectionMerging *= generalTracks



