import FWCore.ParameterSet.Config as cms

softPFMuonsTagInfos = cms.EDProducer("SoftPFMuonTagInfoProducer",
  jets              = cms.InputTag("ak4PFJetsCHS"),
  muons             = cms.InputTag("muons"),
  primaryVertex     = cms.InputTag("offlinePrimaryVertices"),
  muonPt            = cms.double(2.),
  muonSIPsig           = cms.double(200.),
  filterIpsig          = cms.double(4.),
  filterRatio1      = cms.double(0.4),
  filterRatio2      = cms.double(0.7),
  filterPromptMuons = cms.bool(False)
)

from Configuration.Eras.Modifier_phase2_timing_cff import phase2_timing
phase2_timing.toModify(
    softPFMuonsTagInfos,
    primaryVertex = cms.InputTag("offlinePrimaryVertices4D"),
)
