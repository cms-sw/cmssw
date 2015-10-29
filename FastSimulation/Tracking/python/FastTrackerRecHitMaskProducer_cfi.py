import FWCore.ParameterSet.Config as cms

fastTrackerRecHitMaskProducer = cms.EDProducer(
    "FastTrackerRecHitMaskProducer",
    minNumberOfLayersWithMeasBeforeFiltering_ = cms.int32(0),
    trackQuality = cms.string("hightPurity"),
    trajectories = cms.InputTag("generalTracks"),
    recHits = cms.InputTag("siTrackerGaussianSmearingRecHits"),
    )

def maskProducerFromClusterRemover(clusterRemover):
    maskProducer = fastTrackerRecHitMaskProducer.clone(
        minNumberOfLayersWithMeasBeforeFiltering = clusterRemover.minNumberOfLayersWithMeasBeforeFiltering,
        TrackQuality = clusterRemover.TrackQuality,
        trajectories = clusterRemover.trajectories,
        )
    if(hasattr(clusterRemover,"trackClassifier")):
        maskProducer.trackClassifier = clusterRemover.trackClassifier
    if(hasattr(clusterRemover,"oldClusterRemovalInfo")):
        maskProducer.oldHitRemovalInfo = cms.InputTag(clusterRemover.oldClusterRemovalInfo.getModuleLabel().replace("Clusters","Masks"))
    return maskProducer

