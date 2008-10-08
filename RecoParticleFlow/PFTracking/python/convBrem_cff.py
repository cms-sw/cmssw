import FWCore.ParameterSet.Config as cms

from RecoParticleFlow.PFTracking.convBremSeeds_cfi import *


##CLUSTERS
gsClusters = cms.EDFilter("TrackClusterRemover",
    oldClusterRemovalInfo = cms.InputTag("fourthClusters"),
    trajectories = cms.InputTag("fourthWithMaterialTracks"),
    pixelClusters = cms.InputTag("fourthClusters"),
    stripClusters = cms.InputTag("fourthClusters"),
    Common = cms.PSet(
        maxChi2 = cms.double(30.0)
    )
)

##PIXEL HITS
import RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi
gsPixelRecHits = RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi.siPixelRecHits.clone()
gsPixelRecHits.src = 'gsClusters:'
##STRIP HITS
import RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi
gsStripRecHits = RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi.siStripMatchedRecHits.clone()
gsStripRecHits.ClusterProducer = 'gsClusters'

##TK CANDIDATES
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
convTkCand = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone()
convTkCand.SeedProducer = 'convBremSeeds'
convTkCand.TrajectoryBuilder = 'convTrajectoryBuilder'


##TRACKS
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
convTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
convTracks.src = 'convTkCand'

##TRAJECTORY BUILDER
import RecoTracker.CkfPattern.CkfTrajectoryBuilderESProducer_cfi
convTrajectoryBuilder = RecoTracker.CkfPattern.CkfTrajectoryBuilderESProducer_cfi.CkfTrajectoryBuilder.clone()
convTrajectoryBuilder.ComponentName = 'convTrajectoryBuilder'
convTrajectoryBuilder.trajectoryFilterName = 'convTrajectoryFilter'
convTrajectoryBuilder.MeasurementTrackerName = 'convMeasurementTracker'


##TRAJECTORY FILTER
import TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi
convTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.clone()
convTrajectoryFilter.ComponentName = 'convTrajectoryFilter'
convTrajectoryFilter.filterPset.maxLostHits = 0
convTrajectoryFilter.filterPset.minimumNumberOfHits = 3

##MEASUREMENT TRACKER
import RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi
convMeasurementTracker = RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi.MeasurementTracker.clone()
convMeasurementTracker.ComponentName = 'convMeasurementTracker'
convMeasurementTracker.pixelClusterProducer = 'gsClusters'
convMeasurementTracker.stripClusterProducer = 'gsClusters'

from FastSimulation.ParticlePropagator.MagneticFieldMapESProducer_cfi import*
from FastSimulation.TrackerSetup.TrackerInteractionGeometryESProducer_cfi import*
convBrem=cms.Sequence(gsClusters*
                      gsPixelRecHits*
                      gsStripRecHits*
                      convBremSeeds*
                      convTkCand*
                      convTracks)
