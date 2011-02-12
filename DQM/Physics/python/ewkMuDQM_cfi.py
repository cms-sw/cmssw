import FWCore.ParameterSet.Config as cms

# DQM monitor module for EWK-WMuNu
ewkMuDQM = cms.EDAnalyzer("EwkMuDQM",
      # Input collections ->
      TrigTag = cms.untracked.InputTag("TriggerResults::HLT"),
      MuonTag = cms.untracked.InputTag("muons"),
      METTag = cms.untracked.InputTag("pfMet"),
      METIncludesMuons = cms.untracked.bool(True),
      JetTag = cms.untracked.InputTag("ak5CaloJets"),


      # Main cuts ->
      MuonTrig = cms.untracked.string("HLT_Mu11"),
      UseTrackerPt = cms.untracked.bool(True),
      PtCut = cms.untracked.double(20.0),
      EtaCut = cms.untracked.double(2.1),
      IsRelativeIso = cms.untracked.bool(True),
      IsCombinedIso = cms.untracked.bool(True),
      IsoCut03 = cms.untracked.double(0.15),
      MtMin = cms.untracked.double(50.0),
      MtMax = cms.untracked.double(999.0),
      MetMin = cms.untracked.double(-999999.),
      MetMax = cms.untracked.double(999999.),
      AcopCut = cms.untracked.double(999.),

      # Muon quality cuts ->
      DxyCut = cms.untracked.double(0.2),
      NormalizedChi2Cut = cms.untracked.double(10.),
      TrackerHitsCut = cms.untracked.int32(11),
      IsAlsoTrackerMuon = cms.untracked.bool(True),

      # To suppress Zmm ->
      PtThrForZ1 = cms.untracked.double(20.0),
      PtThrForZ2 = cms.untracked.double(10.0),

      # To further suppress ttbar ->
      EJetMin = cms.untracked.double(40.),
      NJetMax = cms.untracked.int32(999999)
)
