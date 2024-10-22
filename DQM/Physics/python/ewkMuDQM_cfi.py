import FWCore.ParameterSet.Config as cms

# DQM monitor module for EWK-WMuNu
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
ewkMuDQM = DQMEDAnalyzer('EwkMuDQM',

      stageL1Trigger = cms.uint32(1),

      # Input collections ->
      TrigTag = cms.untracked.InputTag("TriggerResults::HLT"),
      MuonTag = cms.untracked.InputTag("muons"),
      METTag = cms.untracked.InputTag("pfMet"),
      JetTag = cms.untracked.InputTag("ak4PFJets"),
      VertexTag= cms.untracked.InputTag("offlinePrimaryVertices"),
      TrigPathNames = cms.untracked.vstring("HLT_Mu","HLT_DoubleMu","HLT_IsoMu"),

      # Main cuts ->
      PtCut = cms.untracked.double(25.0),
      EtaCut = cms.untracked.double(2.1),
      IsRelativeIso = cms.untracked.bool(True),
      IsCombinedIso = cms.untracked.bool(True),
      IsoCut03 = cms.untracked.double(0.10),
      MtMin = cms.untracked.double(50.0),
      MtMax = cms.untracked.double(9999999),
      MetMin = cms.untracked.double(-999999.),
      MetMax = cms.untracked.double(999999.),
      AcopCut = cms.untracked.double(999.),

      # Muon quality cuts ->
      DxyCut = cms.untracked.double(0.2), # dxy < 0.2 cm (cosmics)
      NormalizedChi2Cut = cms.untracked.double(10.), # chi2/ndof < 10.
      TrackerHitsCut = cms.untracked.int32(11),  # Hits in inner track > 10
      PixelHitsCut = cms.untracked.int32(1),  # Pixel Hits  > 0
      MuonHitsCut = cms.untracked.int32(1),  # Valid Muon Hits  > 0
      IsAlsoTrackerMuon = cms.untracked.bool(True),
      NMatchesCut = cms.untracked.int32(2),  # At least 2 Chambers matched with segments

      # To suppress Zmm ->
      PtThrForZ1 = cms.untracked.double(20.0),
      PtThrForZ2 = cms.untracked.double(10.0),

      # To further suppress ttbar ->
      EJetMin = cms.untracked.double(40.),
      NJetMax = cms.untracked.int32(999999)
)
from Configuration.Eras.Modifier_stage2L1Trigger_cff import stage2L1Trigger
stage2L1Trigger.toModify(ewkMuDQM, stageL1Trigger = 2)
