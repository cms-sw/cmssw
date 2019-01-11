import FWCore.ParameterSet.Config as cms

TOFPIDProducer = cms.EDProducer( "TOFPIDProducer",
  tracksSrc = cms.InputTag("generalTracks"),
  t0Src = cms.InputTag("trackExtenderWithMTD:generalTrackt0"),
  tmtdSrc = cms.InputTag("trackExtenderWithMTD:generalTracktmtd"),
  sigmatSrc = cms.InputTag("trackExtenderWithMTD:generalTracksigmatmtd"),
  pathLengthSrc = cms.InputTag("trackExtenderWithMTD:generalTrackPathLength"),
  pSrc = cms.InputTag("trackExtenderWithMTD:generalTrackp"),
  vtxsSrc = cms.InputTag("unsortedOfflinePrimaryVertices4DnoPID"),
  vtxMaxSigmaT = cms.double(0.060),
  maxDz = cms.double(0.1),
  maxDtSignificance = cms.double(999.),
  minProbHeavy = cms.double(0.75),
)
