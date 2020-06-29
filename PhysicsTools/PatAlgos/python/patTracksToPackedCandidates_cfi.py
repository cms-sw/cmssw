import FWCore.ParameterSet.Config as cms

patTracksToPackedCandidates = cms.EDProducer('PATTracksToPackedCandidates',
  srcTracks = cms.InputTag('hiConformalPixelTracks'),
  srcPrimaryVertices = cms.InputTag('offlineSlimmedPrimaryVertices'),
  dzSigCut = cms.double(10),
  dxySigCut = cms.double(25),
  dzSigHP = cms.double(7), #to set HighPurity flag for pixel tracks
  dxySigHP = cms.double(20), #to set HighPurity flag for pixel tracks
  ptMax = cms.double(1), #max pT for pixel tracks - above this use general tracks instead
  ptMin = cms.double(0.3), #min pT for pixel tracks
  resetHP = cms.bool(True), #pixel tracks do not have HP flag set. Use False if does not want to reset HP flag
  covarianceVersion = cms.int32(0), #so far: 0 is Phase0, 1 is Phase1
  covarianceSchema = cms.int32(520), #use less accurate schema - reduce size of collection
  mightGet = cms.optional.untracked.vstring
)
