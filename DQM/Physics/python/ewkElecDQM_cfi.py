import FWCore.ParameterSet.Config as cms

# DQM monitor module for EWK-WMuNu
ewkElecDQM = cms.EDAnalyzer("EwkElecDQM",
      # Input collections ->
      TrigTag = cms.untracked.InputTag("TriggerResults::HLT"),
#      MuonTag = cms.untracked.InputTag("muons"),
      ElecTag = cms.untracked.InputTag("gedGsfElectrons"),
      METTag = cms.untracked.InputTag("pfMet"),
      METIncludesMuons = cms.untracked.bool(False),
      JetTag = cms.untracked.InputTag("ak4PFJets"),
#      JetTag = cms.untracked.InputTag("ak4CaloJets"),
#      JetTag = cms.untracked.InputTag("sisCone5CaloJets"),
      VertexTag= cms.untracked.InputTag("offlinePrimaryVertices"),
      BeamSpotTag = cms.untracked.InputTag("offlineBeamSpot"),

      # Main cuts ->
#      MuonTrig = cms.untracked.string("HLT_Mu9"),
       ElecTrig = cms.untracked.vstring("HLT_Ele","HLT_DoubleEle","HLT_DoublePhoton","HLT_Photon","HLT_L1SingleEG"),
#      ElecTrig = cms.untracked.vstring("*"),
#      UseTrackerPt = cms.untracked.bool(True), # ??
#      PtCut = cms.untracked.double(25.0),
      PtCut = cms.untracked.double(10.0),
#      EtaCut = cms.untracked.double(2.1),
      EtaCut = cms.untracked.double(2.4),
      SieieBarrel = cms.untracked.double(0.01),
      SieieEndcap = cms.untracked.double(0.028),
      DetainBarrel = cms.untracked.double(0.0071),
      DetainEndcap = cms.untracked.double(0.0066),
#      IsRelativeIso = cms.untracked.bool(True),
#      IsCombinedIso = cms.untracked.bool(False),
#      IsoCut03 = cms.untracked.double(0.1),
      EcalIsoCutBarrel = cms.untracked.double(5.7),
      EcalIsoCutEndcap = cms.untracked.double(5.0),
      HcalIsoCutBarrel = cms.untracked.double(8.1),
      HcalIsoCutEndcap = cms.untracked.double(3.4),
      TrkIsoCutBarrel = cms.untracked.double(7.2),
      TrkIsoCutEndcap = cms.untracked.double(5.1),
      MtMin = cms.untracked.double(-999999),
      MtMax = cms.untracked.double(999999.0),
      MetMin = cms.untracked.double(-999999.),
      MetMax = cms.untracked.double(999999.),
#      AcopCut = cms.untracked.double(2.),

#      # Muon quality cuts ->
#      DxyCut = cms.untracked.double(0.2),
#      NormalizedChi2Cut = cms.untracked.double(10.),
#      TrackerHitsCut = cms.untracked.int32(11),
#      IsAlsoTrackerMuon = cms.untracked.bool(True),

#      # To suppress Zmm ->
#      PtThrForZ1 = cms.untracked.double(20.0),
#      PtThrForZ2 = cms.untracked.double(10.0),

      # To further suppress ttbar ->
      EJetMin = cms.untracked.double(30.),
      NJetMax = cms.untracked.int32(999999),

      # PU dependence
      PUMax = cms.untracked.uint32(60),
      PUBinCount = cms.untracked.uint32(12)  # Bin size PUMax/PUBinCount
)
