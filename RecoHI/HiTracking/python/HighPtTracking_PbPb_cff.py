import FWCore.ParameterSet.Config as cms

### common stuff
from TrackingTools.TransientTrack.TransientTrackBuilder_cfi import * #also includes global tracking geometry
from RecoTracker.TransientTrackingRecHit.TTRHBuilders_cff import *
from RecoLocalTracker.SiPixelRecHits.PixelCPEESProducers_cff import * #cluster parameter estimator producer
from RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi import *

### pixel primary vertices
from RecoHI.HiTracking.HIPixelVertices_cff import *

### pixel triplet seeds with vertex constraint
from RecoHI.HiTracking.HIPixelTripletSeeds_cff import *

### primary ckf track candidates
from RecoHI.HiTracking.HICkfTrackCandidates_cff import *

### final track fit
from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cff import *
hiGlobalPrimTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone(
	src = 'hiPrimTrackCandidates',
	useHitsSplitting=True,
        AlgorithmName = 'iter0'
)

### track quality cuts
from RecoHI.HiTracking.HISelectedTracks_cfi import *

### paths
hiBasicTracking = cms.Sequence(hiPixelVertices
                                * hiPrimSeeds
                                * hiPrimTrackCandidates
                                * hiGlobalPrimTracks
                                * hiTracksWithQuality
                                )










