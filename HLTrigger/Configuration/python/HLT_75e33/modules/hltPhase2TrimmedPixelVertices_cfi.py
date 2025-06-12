import FWCore.ParameterSet.Config as cms

hltPhase2TrimmedPixelVertices = cms.EDProducer("PixelVertexCollectionTrimmer",
    src             = cms.InputTag("hltPhase2PixelVertices"),
    maxVtx          = cms.uint32(300),
    fractionSumPt2  = cms.double(0.3),
    minSumPt2       = cms.double(0.0),
    PVcomparer      = cms.PSet(
        refToPSet_ = cms.string("pSetPvClusterComparerForIT"),
    )
)
