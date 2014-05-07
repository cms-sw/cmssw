import FWCore.ParameterSet.Config as cms
unpackedTracksAndVertices = cms.EDProducer('PATTrackAndVertexUnpacker',
 slimmedVertices = cms.InputTag("offlineSlimmedPrimaryVertices"),
 slimmedSecondaryVertices = cms.InputTag("slimmedSecondaryVertices"),
 additionalTracks= cms.InputTag("lostTracks"),
 packedCandidates = cms.InputTag("packedPFCandidates")
)

