from ElectroWeakAnalysis.WMuNu.wmunusProducer_cfi import *
# Paths for WMuNuSelector filtering of events

selcorMet = cms.EDFilter("WMuNuSelector",
      # Fill Basc Histograms? ->
      plotHistograms = cms.untracked.bool(False),
      saveNTuple = cms.untracked.bool(False),

      # Input collections ->
      MuonTag = cms.untracked.InputTag("muons"),
      TrigTag = cms.untracked.InputTag("TriggerResults::HLT"), # CAREFUL --> In Summer08 and in data, "HLT". In Spring10 --> REDIGI, etc, etc
      JetTag = cms.untracked.InputTag("ak5PFJets"),
      WMuNuCollectionTag = cms.untracked.InputTag("corMetWMuNus"),
      VertexTag = cms.untracked.InputTag("offlinePrimaryVertices"),

      # Preselection! 
      MuonTrig = cms.untracked.vstring("HLT_Mu9","HLT_Mu11","HLT_Mu15_v1"), 
      PtThrForZ1 = cms.untracked.double(20.0),
      PtThrForZ2 = cms.untracked.double(10.0),
      EJetMin = cms.untracked.double(40.),
      NJetMax = cms.untracked.int32(999999),
      MinVertices = cms.untracked.int32(0),
      MaxVertices = cms.untracked.int32(999),

      # Main cuts ->
      PtCut = cms.untracked.double(25.0), # Edited for Moriond
      EtaCut = cms.untracked.double(2.1),
      IsRelativeIso = cms.untracked.bool(True),
      IsCombinedIso = cms.untracked.bool(True), 
      IsoCut03 = cms.untracked.double(0.10),    # Edited for Moriond
      MtMin = cms.untracked.double(50.0),
      MtMax = cms.untracked.double(999999.),
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

      # Select only W-, W+ ( default is all Ws)  
      SelectByCharge=cms.untracked.int32(0)

)

selpfMet = selcorMet.clone()
selpfMet.WMuNuCollectionTag = cms.untracked.InputTag("pfMetWMuNus")

seltcMet = selcorMet.clone()
seltcMet.WMuNuCollectionTag = cms.untracked.InputTag("tcMetWMuNus")

selectCaloMetWMuNus = cms.Sequence(corMetWMuNus+selcorMet)

selectPfMetWMuNus = cms.Sequence(pfMetWMuNus+selpfMet)

selectTcMetWMuNus = cms.Sequence(tcMetWMuNus+seltcMet)


