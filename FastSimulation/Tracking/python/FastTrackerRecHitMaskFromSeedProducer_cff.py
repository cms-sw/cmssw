import FWCore.ParameterSet.Config as cms

from FastSimulation.Tracking.fastTrackerRecHitMaskFromSeedProducer_cfi import fastTrackerRecHitMaskFromSeedProducer

def maskProducerFromSeedClusterRemover(seedClusterRemover):
    maskProducer = fastTrackerRecHitMaskFromSeedProducer.clone(
        trajectories = seedClusterRemover.trajectories,
        )
    if(hasattr(seedClusterRemover,"oldClusterRemovalInfo")):
        maskProducer.oldHitRemovalInfo = cms.InputTag(seedClusterRemover.oldClusterRemovalInfo.getModuleLabel().replace("Clusters","Masks"))
    return maskProducer
