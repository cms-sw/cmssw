import FWCore.ParameterSet.Config as cms

from RecoPixelVertexing.PixelLowPtUtilities.common_cff import *
from RecoPixelVertexing.PixelLowPtUtilities.firstStep_cff import *
from RecoPixelVertexing.PixelLowPtUtilities.secondStep_cff import *
from RecoPixelVertexing.PixelLowPtUtilities.thirdStep_cff import *

ckfBaseTrajectoryFilterForMinBias.filterPset.minimumNumberOfHits = 3
ckfBaseTrajectoryFilterForMinBias.filterPset.minPt           = 0.075

from RecoVZero.VZeroFinding.VZeros_cff import *

#allTracks = cms.EDFilter("TrackListCombiner",
#    trackProducers = cms.vstring('globalPrimTracks', 
#        'globalSecoTracks')
#)

allTracks = cms.EDFilter("TrackListCombiner",
    trackProducers = cms.vstring('globalPrimTracks', 
        'globalSecoTracks')
)

#allTracks = cms.EDFilter("SimpleTrackListMerger",
    # minimum shared fraction to be called duplicate
#    ShareFrac = cms.double(1.00),
#    MinPT = cms.double(0.0),
#    Epsilon = cms.double(-0.001),
#    MaxNormalizedChisq = cms.double(999999.),
#    MinFound = cms.int32(0),
#    TrackProducer1 = cms.string('globalPrimTracks')
#    TrackProducer2 = cms.string('globalSecoTracks'),
#)

firstStep = cms.Sequence(pixel3ProtoTracks*pixelVertices*pixel3PrimTracks*primSeeds*primTrackCandidates*globalPrimTracks)
secondStep = cms.Sequence(secondClusters*secondPixelRecHits*secondStripRecHits*pixelSecoTracks*secoSeeds*secoTrackCandidates*globalSecoTracks)
thirdStep = cms.Sequence(thirdClusters*thirdPixelRecHits*thirdStripRecHits*pixel2PrimTracks*tertSeeds*tertTrackCandidates*globalTertTracks)
minBiasTracking = cms.Sequence(firstStep*secondStep*allTracks)

