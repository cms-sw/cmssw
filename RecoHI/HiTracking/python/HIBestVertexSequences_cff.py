import FWCore.ParameterSet.Config as cms

# sort by number of tracks and keep the best
hiBestAdaptiveVertex = cms.EDFilter("HIBestVertexSelection",
    src = cms.InputTag("hiPixelAdaptiveVertex"),
	maxNumber = cms.uint32(1)
)

# select best of precise vertex, fast vertex, and beamspot
hiSelectedPixelVertex = cms.EDProducer("HIBestVertexProducer",
    beamSpotLabel = cms.InputTag("offlineBeamSpot"),
    adaptiveVertexCollection = cms.InputTag("hiBestAdaptiveVertex"),
    medianVertexCollection = cms.InputTag("hiPixelMedianVertex"),
    useFinalAdaptiveVertexCollection = cms.bool(False),
)

# best vertex sequence
bestHiVertexTask = cms.Task( hiBestAdaptiveVertex , hiSelectedPixelVertex ) # vertexing run BEFORE tracking

from RecoHI.HiTracking.HIPixelAdaptiveVertex_cfi import *
hiOfflinePrimaryVertices=hiPixelAdaptiveVertex.clone( # vertexing run AFTER tracking
    TrackLabel = cms.InputTag("hiGeneralTracks"),
                                       
    TkFilterParameters = cms.PSet(
        algorithm = cms.string('filterWithThreshold'),
        maxNormalizedChi2 = cms.double(5.0),
        minPixelLayersWithHits=cms.int32(3),    #0 missing pix hit (Run 1 pixels)
        minSiliconLayersWithHits = cms.int32(5),#at least 8 (3pix+5strip) hits total
        maxD0Significance = cms.double(3.0),    #default is 5.0 in pp; 3.0 here suppresses split vtxs
        minPt = cms.double(0.0),
        maxEta = cms.double(100.),               
        trackQuality = cms.string("any"),
        numTracksThreshold = cms.int32(2)
    )
)
from Configuration.Eras.Modifier_trackingPhase1_cff import trackingPhase1
trackingPhase1.toModify(hiOfflinePrimaryVertices,
	TkFilterParameters = dict (minPixelLayersWithHits = 4, minSiliconLayersWithHits = 4) 
	#Phase 1 requires 8 hits total, but 4 from pixels, 4 from strips now instead of 3 and 5
)

hiBestOfflinePrimaryVertex = cms.EDFilter("HIBestVertexSelection",
    src = cms.InputTag("hiOfflinePrimaryVertices"),
    maxNumber = cms.uint32(1)
)
# select best of precise vertex, fast vertex, and beamspot
hiSelectedVertex = hiSelectedPixelVertex.clone(
    useFinalAdaptiveVertexCollection = cms.bool(True),
    finalAdaptiveVertexCollection = cms.InputTag("hiBestOfflinePrimaryVertex")
)
bestFinalHiVertexTask = cms.Task(hiOfflinePrimaryVertices , hiBestOfflinePrimaryVertex , hiSelectedVertex )
