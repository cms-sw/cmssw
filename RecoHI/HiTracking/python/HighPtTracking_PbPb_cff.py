import FWCore.ParameterSet.Config as cms

### common imports
from TrackingTools.Configuration.TrackingTools_cff import *
from TrackingTools.TransientTrack.TransientTrackBuilder_cfi import * #also includes global tracking geometry
from RecoTracker.TkSeedingLayers.TTRHBuilderWithoutAngle4PixelTriplets_cfi import * #StripCPE = 'Fake'
from RecoLocalTracker.SiPixelRecHits.PixelCPEESProducers_cff import * #cluster parameter estimator producer
from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *

### pixel triplets for vertexing
import RecoHI.HiTracking.PixelProtoTracks_cfi
pixel3ProtoTracks = RecoHI.HiTracking.PixelProtoTracks_cfi.hiProtoTracks.clone()
pixel3ProtoTracks.passLabel = 'Pixel triplet tracks for vertexing'
pixel3ProtoTracks.RegionFactoryPSet.RegionPSet.ptMin = 0.7
pixel3ProtoTracks.RegionFactoryPSet.RegionPSet.originRadius = 0.1

### pixel vertices
from RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi import *
#from RecoPixelVertexing.PixelVertexFinding.PixelVertexes_cfi import *  #divisive vertex finder
from RecoHI.HiTracking.PixelVertices_cfi import *  #median vertex producer
pixel3Vertices.TrackCollection = 'pixel3ProtoTracks'

### pixel triplets with vertex
import RecoPixelVertexing.PixelLowPtUtilities.AllPixelTracks_cfi
pixel3PrimTracks = RecoPixelVertexing.PixelLowPtUtilities.AllPixelTracks_cfi.allPixelTracks.clone()
pixel3PrimTracks.passLabel = 'Pixel triplet primary tracks with vertex constraint'
pixel3PrimTracks.RegionFactoryPSet.RegionPSet.ptMin = 1.5
pixel3PrimTracks.RegionFactoryPSet.RegionPSet.originRadius = 0.2
pixel3PrimTracks.RegionFactoryPSet.RegionPSet.useFoundVertices = True
pixel3PrimTracks.OrderedHitsFactoryPSet.GeneratorPSet.checkClusterShape = False # don't use low-pt cluster shape filter
pixel3PrimTracks.CleanerPSet.ComponentName= "PixelTrackCleanerBySharedHits" # if two or more shared hits, discard lower pt track

### pixel seeds
import RecoPixelVertexing.PixelLowPtUtilities.TrackSeeds_cfi
primSeeds = RecoPixelVertexing.PixelLowPtUtilities.TrackSeeds_cfi.pixelTrackSeeds.clone()
primSeeds.InputCollection = 'pixel3PrimTracks'

### base trajectory filter
from RecoTracker.CkfPattern.CkfTrackCandidates_cff import * #also includes both trajectory builders
from TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cff import *
ckfBaseTrajectoryFilter.filterPset.minimumNumberOfHits = 6
ckfBaseTrajectoryFilter.filterPset.minPt = 2.0 #default is 0.9

### trajectory builder
GroupedCkfTrajectoryBuilder.maxCand = 5
GroupedCkfTrajectoryBuilder.intermediateCleaning = False
GroupedCkfTrajectoryBuilder.alwaysUseInvalidHits = False
MaterialPropagator.Mass = 0.139 #pion (default is muon)
OppositeMaterialPropagator.Mass = 0.139

### primary track candidates
from TrackingTools.TrajectoryCleaning.TrajectoryCleanerBySharedSeeds_cfi import *
from RecoTracker.CkfPattern.CkfTrackCandidates_cff import * #also includes both trajectory builders
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
primTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone()
primTrackCandidates.TrajectoryCleaner = 'TrajectoryCleanerBySharedSeeds'
primTrackCandidates.TrajectoryBuilder = 'CkfTrajectoryBuilder' #instead of GroupedCkfTrajectoryBuilder?
#primTrackCandidates.SeedProducer = 'primSeeds'
primTrackCandidates.src = 'primSeeds' #change for 3_1_x
primTrackCandidates.RedundantSeedCleaner = 'none'
primTrackCandidates.doSeedingRegionRebuilding = False 

### final track fit
from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cff import *
from TrackingTools.TrackFitters.KFFittingSmootherESProducer_cfi import *
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
globalPrimTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
globalPrimTracks.src = 'primTrackCandidates'
globalPrimTracks.TrajectoryInEvent = True
globalPrimTracks.useHitsSplitting = True #Vasu's modification
globalPrimTracks.Fitter = 'KFFittingSmoother' #Vasu's modification

### paths
heavyIonTracking = cms.Sequence(pixel3ProtoTracks*pixel3Vertices*pixel3PrimTracks*primSeeds*primTrackCandidates*globalPrimTracks)











