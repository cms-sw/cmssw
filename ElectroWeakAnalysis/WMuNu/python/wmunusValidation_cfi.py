import FWCore.ParameterSet.Config as cms

wmnVal_corMet = cms.EDFilter("WMuNuValidator",
      # Fast selection flag (no histograms or book-keeping) ->
      FastOption = cms.untracked.bool(False),

      # Input collections ->
      TrigTag = cms.untracked.InputTag("TriggerResults::HLT"),# CAREFUL --> In Summer08 and in data, "HLT". In Spring10 --> REDIGI
      MuonTag = cms.untracked.InputTag("muons"),
      METTag = cms.untracked.InputTag("corMetGlobalMuons"),
      METIncludesMuons = cms.untracked.bool(True),
      JetTag = cms.untracked.InputTag("ak5CaloJets"),  # CAREFUL --> If you run on Summer09 MC, this was called "antikt5CaloJets"
      
      # Main cuts ->
      MuonTrig = cms.untracked.string("HLT_Mu9"),
      PtCut = cms.untracked.double(20.0),
      EtaMinCut = cms.untracked.double(-2.1),
      EtaMaxCut = cms.untracked.double(2.1),
      IsRelativeIso = cms.untracked.bool(True),
      IsCombinedIso = cms.untracked.bool(True),# Combined Iso used (cut at 0.15 as it is equivalent for   
      IsoCut03 = cms.untracked.double(0.15),   # signal to 0.1 with track-iso)
      MtMin = cms.untracked.double(50.0),
      MtMax = cms.untracked.double(9999999.),
      MetMin = cms.untracked.double(-999999.),
      MetMax = cms.untracked.double(999999.),
      AcopCut = cms.untracked.double(999),     # Remember to take this out if you are looking for High-Pt Bosons! (V+Jets)

      # Muon quality cuts ->
      DxyCut = cms.untracked.double(0.2), # dxy < 0.2 cm (cosmics)
      NormalizedChi2Cut = cms.untracked.double(10.), # chi2/ndof < 10 
      TrackerHitsCut = cms.untracked.int32(11),  # Hits in inner track > 10
      PixelHitsCut = cms.untracked.int32(1),  # Pixel Hits  > 0 
      MuonHitsCut = cms.untracked.int32(1),  # Valid Muon Hits  > 0 
      IsAlsoTrackerMuon = cms.untracked.bool(True),
      NMatchesCut = cms.untracked.int32(2),  # Number of Chambers matched with segments >= 2
 
      # To suppress Zmm ->
      PtThrForZ1 = cms.untracked.double(20.0),
      PtThrForZ2 = cms.untracked.double(10.0),
      
      # To further suppress ttbar ->
      EJetMin = cms.untracked.double(40.),
      NJetMax = cms.untracked.int32(999999)
)

wmnVal_pfMet = wmnVal_corMet.clone()
wmnVal_pfMet.METTag = cms.untracked.InputTag("pfMet")

wmnVal_tcMet = wmnVal_corMet.clone()
wmnVal_tcMet.METTag = cms.untracked.InputTag("tcMet")



wmunuVal = cms.Sequence(wmnVal_corMet+wmnVal_tcMet+wmnVal_pfMet)
