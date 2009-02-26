import FWCore.ParameterSet.Config as cms

import RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi
from TrackingTools.GeomPropagators.AnalyticalPropagator_cfi import *
from RecoPixelVertexing.PixelVertexFinding.PixelVertexes_cff import *
from RecoLocalTracker.SiPixelRecHits.PixelCPEESProducers_cff import *
from RecoTracker.CkfPattern.CkfTrackCandidates_cff import *
from RecoPixelVertexing.PixelLowPtUtilities.MinBiasCkfTrajectoryFilterESProducer_cfi import *
from TrackingTools.TrajectoryCleaning.TrajectoryCleanerBySharedSeeds_cfi import *
from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cff import *
GlobalTrackingGeometryESProducer = cms.ESProducer("GlobalTrackingGeometryESProducer")

TransientTrackBuilderESProducer = cms.ESProducer("TransientTrackBuilderESProducer",
    ComponentName = cms.string('TransientTrackBuilder'),
)

#ttrhbwr.ComputeCoarseLocalPositionFromDisk = cms.bool(True)

BPixError = cms.PSet(
    useErrorsFromParam = cms.untracked.bool(True),
    hitErrorRPhi = cms.double(0.0027),
    hitErrorRZ = cms.double(0.006)
)
FPixError = cms.PSet(
    useErrorsFromParam = cms.untracked.bool(True),
    hitErrorRPhi = cms.double(0.0051),
    hitErrorRZ = cms.double(0.0036)
)

GroupedCkfTrajectoryBuilder.maxCand = 5
GroupedCkfTrajectoryBuilder.intermediateCleaning = False
GroupedCkfTrajectoryBuilder.alwaysUseInvalidHits = False
GroupedCkfTrajectoryBuilder.trajectoryFilterName = 'MinBiasCkfTrajectoryFilter'
MaterialPropagator.Mass = 0.139
OppositeMaterialPropagator.Mass = 0.139

