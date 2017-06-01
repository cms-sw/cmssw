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
    medianVertexCollection = cms.InputTag("hiPixelMedianVertex")
)

# best vertex sequence
bestHiVertex = cms.Sequence( hiBestAdaptiveVertex * hiSelectedPixelVertex ) # vertexing run BEFORE tracking

from RecoHI.HiTracking.HIPixelAdaptiveVertex_cfi import *
hiOfflinePrimaryVertices=hiPixelAdaptiveVertex.clone( # vertexing run AFTER tracking
    TrackLabel = cms.InputTag("hiGeneralTracks"),
                                       
    TkFilterParameters = cms.PSet(
        algorithm = cms.string('filterWithThreshold'),
        maxNormalizedChi2 = cms.double(5.0),
        minPixelLayersWithHits=cms.int32(3),    #0 missing pix hit
        minSiliconLayersWithHits = cms.int32(5),#at least 8 hits total
        maxD0Significance = cms.double(3.0),    #default 5.0, suppresses split vtxs
        minPt = cms.double(0.0),
        maxEta = cms.double(100.),               
        trackQuality = cms.string("any"),
        numTracksThreshold = cms.int32(2)
    )
)
from Configuration.Eras.Modifier_trackingPhase1_cff import trackingPhase1
trackingPhase1.toModify(hiOfflinePrimaryVertices,
TkFilterParameters = cms.PSet(
        algorithm = cms.string('filterWithThreshold'),
        maxNormalizedChi2 = cms.double(5.0),
        minPixelLayersWithHits=cms.int32(4),    #0 missing pix hit in phase1
        minSiliconLayersWithHits = cms.int32(4),#at least 8 hits total
        maxD0Significance = cms.double(3.0),    #default 5.0, suppresses split vtxs
        minPt = cms.double(0.0),
        maxEta = cms.double(100.),
        trackQuality = cms.string("any"),
        numTracksThreshold = cms.int32(2)
    )
)


hiBestOfflinePrimaryVertex = cms.EDFilter("HIBestVertexSelection",
    src = cms.InputTag("hiOfflinePrimaryVertices"),
    maxNumber = cms.uint32(1)
)
# select best of precise vertex, fast vertex, and beamspot
hiSelectedVertex = cms.EDProducer("HIBestVertexProducer",
    beamSpotLabel = cms.InputTag("offlineBeamSpot"),
    medianVertexCollection = cms.InputTag("hiPixelMedianVertex"),
    adaptiveVertexCollection = cms.InputTag("hiBestAdaptiveVertex"),
    useFinalAdapativeVertexCollection = cms.bool(True),
    finalAdaptiveVertexCollection = cms.InputTag("hiBestOfflinePrimaryVertex")
)
bestFinalHiVertex = cms.Sequence(hiOfflinePrimaryVertices * hiBestOfflinePrimaryVertex * hiSelectedVertex ) # vertexing run BEFORE tracking
