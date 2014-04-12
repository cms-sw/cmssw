import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiPixelRecHits.PixelCPEESProducers_cff import *
from RecoTracker.CkfPattern.CkfTrackCandidates_cff import *
from RecoPixelVertexing.PixelLowPtUtilities.MinBiasCkfTrajectoryFilterESProducer_cfi import *
from TrackingTools.TrajectoryCleaning.TrajectoryCleanerBySharedSeeds_cfi import *
from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cff import *

# Global tracking geometry
GlobalTrackingGeometryESProducer = cms.ESProducer("GlobalTrackingGeometryESProducer")

# Transient track builder
TransientTrackBuilderESProducer = cms.ESProducer("TransientTrackBuilderESProducer",
    ComponentName = cms.string('TransientTrackBuilder'),
)

# Trajectory builder
GroupedCkfTrajectoryBuilder.maxCand = 5
GroupedCkfTrajectoryBuilder.intermediateCleaning = False
GroupedCkfTrajectoryBuilder.alwaysUseInvalidHits = False
GroupedCkfTrajectoryBuilder.trajectoryFilterName = 'MinBiasCkfTrajectoryFilter'
GroupedCkfTrajectoryBuilder.inOutTrajectoryFilterName = 'MinBiasCkfTrajectoryFilter'
GroupedCkfTrajectoryBuilder.useSameTrajFilter = cms.bool(True)

# Propagator, pion mass
MaterialPropagator.Mass          = cms.double(0.139)
OppositeMaterialPropagator.Mass  = cms.double(0.139)
RungeKuttaTrackerPropagator.Mass = cms.double(0.139)

#from TrackingTools.TrackFitters.KFFittingSmootherWithOutliersRejectionAndRK_cfi import *
#KFFittingSmootherWithOutliersRejectionAndRK.EstimateCut = cms.double(999999.)
