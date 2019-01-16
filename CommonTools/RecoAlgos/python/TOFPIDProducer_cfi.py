import FWCore.ParameterSet.Config as cms

tofPID = cms.EDProducer( "TOFPIDProducer",
  tracksSrc = cms.InputTag("generalTracks"),
  t0Src = cms.InputTag("trackExtenderWithMTD:generalTrackt0"),
  tmtdSrc = cms.InputTag("trackExtenderWithMTD:generalTracktmtd"),
  sigmat0Src = cms.InputTag("trackExtenderWithMTD:generalTracksigmat0"),
  sigmatmtdSrc = cms.InputTag("trackExtenderWithMTD:generalTracksigmatmtd"),
  pathLengthSrc = cms.InputTag("trackExtenderWithMTD:generalTrackPathLength"),
  pSrc = cms.InputTag("trackExtenderWithMTD:generalTrackp"),
  vtxsSrc = cms.InputTag("unsortedOfflinePrimaryVertices4DnoPID"),
  vtxMaxSigmaT = cms.double(0.03),
  maxDz = cms.double(0.1),
  maxDtSignificance = cms.double(6.),
  minProbHeavy = cms.double(0.75),
)
