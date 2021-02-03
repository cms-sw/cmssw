import FWCore.ParameterSet.Config as cms

ghostTrackVertexRecoBlock = cms.PSet(
    vertexReco = cms.PSet(
        finder = cms.string('gtvr'),
        fitType = cms.string('RefitGhostTrackWithVertices'),
        maxFitChi2 = cms.double(10.0),
        mergeThreshold = cms.double(3.0),
        primcut = cms.double(2.0),
        seccut = cms.double(4.0)
    )
)