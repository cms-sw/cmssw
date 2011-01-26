import FWCore.ParameterSet.Config as cms
# VBTF analysis using a full PF-Based approach (PFIsolation & PFMet)

# Create a new reco::Muon collection with PFLow Iso information
# This is the first version of the producer. Please get cvs co -r V00-02-04 ElectroWeakAnalysis/Utilities 
muonsWithPFIso = cms.EDFilter("MuonWithPFIsoProducer",
        MuonTag = cms.untracked.InputTag("muons")
      , PfTag = cms.untracked.InputTag("particleFlow")
      , UsePfMuonsOnly = cms.untracked.bool(False)
      , TrackIsoVeto = cms.untracked.double(0.01) # This is to be compatible with the standard Muon isolation vetos
      , GammaIsoVeto = cms.untracked.double(0.07) 
      , NeutralHadronIsoVeto = cms.untracked.double(0.1)
)

# Build WMuNu Collection:
pfWMuNus = cms.EDProducer("WMuNuProducer",
      # Input collections ->
      MuonTag = cms.untracked.InputTag("muonsWithPFIso"),
      METTag = cms.untracked.InputTag("pfMet"),
      OnlyHighestPtCandidate = cms.untracked.bool(True) # Only 1 Candidate saved in the event
)

# Select them:
selPFWMuNus = cms.EDFilter("WMuNuSelector",
      # Fill Basc Histograms? ->
      plotHistograms = cms.untracked.bool(False),

      # Input collections ->
      MuonTag = cms.untracked.InputTag("muonsWithPFIso"),
      TrigTag = cms.untracked.InputTag("TriggerResults::HLT"),
      JetTag = cms.untracked.InputTag("ak5PFJets"), # CAREFUL --> If you run on Summer09 MC, this was called "antikt5CaloJets"
      WMuNuCollectionTag = cms.untracked.InputTag("pfWMuNus"),

      # Preselection! 
      MuonTrig = cms.untracked.vstring("HLT_Mu9","HLT_Mu11","HLT_Mu15_v1"),
      PtThrForZ1 = cms.untracked.double(20.0),
      PtThrForZ2 = cms.untracked.double(10.0),
      EJetMin = cms.untracked.double(40.),
      NJetMax = cms.untracked.int32(999999),

      # Main cuts ->
      PtCut = cms.untracked.double(25.0),
      EtaCut = cms.untracked.double(2.1),
      IsRelativeIso = cms.untracked.bool(True),
      IsCombinedIso = cms.untracked.bool(True), #--> Changed default to Combined Iso. A cut in 0.15 is equivalent (for signal)
      IsoCut03 = cms.untracked.double(0.10),    # to a cut in TrackIso in 0.10
      MtMin = cms.untracked.double(50.0),
      MtMax = cms.untracked.double(999999.),
      MetMin = cms.untracked.double(-999999.),
      MetMax = cms.untracked.double(999999.),
      AcopCut = cms.untracked.double(999.),  # Remember to take this out if you are looking for High-Pt Bosons! (V+Jets)

      # Muon quality cuts ->
      DxyCut = cms.untracked.double(0.2),
      NormalizedChi2Cut = cms.untracked.double(10.),
      TrackerHitsCut = cms.untracked.int32(11),
      MuonHitsCut = cms.untracked.int32(1),
      IsAlsoTrackerMuon = cms.untracked.bool(True),
      NMatchesCut = cms.untracked.int32(2),

      # Select only W-, W+ ( default is all Ws)  
      SelectByCharge=cms.untracked.int32(0)

)

selectPFWMuNus = cms.Sequence(muonsWithPFIso+pfWMuNus+selPFWMuNus)

