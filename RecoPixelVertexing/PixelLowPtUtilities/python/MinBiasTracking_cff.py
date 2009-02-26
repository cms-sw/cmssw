import FWCore.ParameterSet.Config as cms

from RecoPixelVertexing.PixelLowPtUtilities.common_cff     import *
from RecoPixelVertexing.PixelLowPtUtilities.firstStep_cff  import *
from RecoPixelVertexing.PixelLowPtUtilities.secondStep_cff import *
from RecoPixelVertexing.PixelLowPtUtilities.thirdStep_cff  import *

ckfBaseTrajectoryFilterForMinBias.filterPset.minimumNumberOfHits = 3
ckfBaseTrajectoryFilterForMinBias.filterPset.minPt           = 0.075

from RecoVZero.VZeroFinding.VZeros_cff import *

######################
# Tracklist combiner
allTracks = cms.EDFilter("TrackListCombiner",
    trackProducers = cms.vstring('globalPrimTracks', 
        'globalSecoTracks')
)

firstStep  = cms.Sequence(pixel3ProtoTracks
                        * pixelVertices
                        * pixel3PrimTracks
                        * primSeeds
                        * primTrackCandidates
                        * globalPrimTracks)

secondStep = cms.Sequence(secondClusters
                        * secondPixelRecHits
                        * secondStripRecHits
                        * pixelSecoTracks
                        * secoSeeds
                        * secoTrackCandidates
                        * globalSecoTracks)

thirdStep  = cms.Sequence( thirdClusters
                         * thirdPixelRecHits
                         * thirdStripRecHits
                         * pixel2PrimTracks
                         * tertSeeds
                         * tertTrackCandidates
                         * globalTertTracks)

minBiasTracking = cms.Sequence(firstStep
                             * secondStep
                             * allTracks)

