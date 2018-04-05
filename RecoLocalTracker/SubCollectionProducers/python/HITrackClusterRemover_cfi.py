import FWCore.ParameterSet.Config as cms

HITrackClusterRemover = cms.EDProducer( "HITrackClusterRemover",
     clusterLessSolution = cms.bool(True),
     trajectories = cms.InputTag("hltHIGlobalPrimTracks"),
     oldClusterRemovalInfo = cms.InputTag( "" ),
     overrideTrkQuals = cms.InputTag( 'hltHIIter0TrackSelection','hiInitialStep' ),
     TrackQuality = cms.string('highPurity'),
     minNumberOfLayersWithMeasBeforeFiltering = cms.int32(0),
     pixelClusters = cms.InputTag("hltHISiPixelClusters"),
     stripClusters = cms.InputTag("hltHITrackingSiStripRawToClustersFacility"),
     Common = cms.PSet(
         maxChi2 = cms.double(9.0),
     ),
     Strip = cms.PSet(
        # Yen-Jie's mod to preserve merged clusters
        maxSize = cms.uint32(2),
        maxChi2 = cms.double(9.0)
     )
)

