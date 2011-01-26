from ElectroWeakAnalysis.WMuNu.wmunusProducer_cfi import *

# Paths for WMuNuSelector filtering of events
# Intended for baseline for specific VBTF taks (like reconstruction efficiency)
# Paths:
# selWMuNu_MuonIDonly --> ONLY MuonID cuts (above 15 GeV)
# selWMuNu_MuonIDAndIso --> "Good" Muons which are also isolated (above 15 GeV)
# selWMuNu_MuonIDAndIsoAndTrigger --> Also passing the HLT_Mu11
# selWMuNu_MuonSelected --> All Muon-related cuts (so all cuts but Met-based ones)
 

selWMuNu_MuonIDonly = cms.EDFilter("WMuNuSelector",
      # Fill Basc Histograms? ->
      plotHistograms = cms.untracked.bool(False),
      saveNTuple = cms.untracked.bool(False),

      # Input collections ->
      MuonTag = cms.untracked.InputTag("muons"),
      TrigTag = cms.untracked.InputTag("TriggerResults::HLT"),
      JetTag = cms.untracked.InputTag(""),  
      WMuNuCollectionTag = cms.untracked.InputTag("pfMetWMuNus"), 

      # Preselection! 
      MuonTrig = cms.untracked.string(""), # No trigger requested
      PtThrForZ1 = cms.untracked.double(1000.0),   # No Z rejection
      PtThrForZ2 = cms.untracked.double(1000.0),

      # Main cuts ->
      PtCut = cms.untracked.double(15.0),  # minimal pt cut, as the QCD Bckg sample the analysis used has a generator level cut in 15
      EtaCut = cms.untracked.double(2.1),
      IsRelativeIso = cms.untracked.bool(True),
      IsCombinedIso = cms.untracked.bool(True), 
      IsoCut03 = cms.untracked.double(9999.),    # No isolation Cut
      MtMin = cms.untracked.double(0.0),         # No MT cut                       
      MtMax = cms.untracked.double(99999.),
      AcopCut = cms.untracked.double(9999.),     # No Acop cut

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

selWMuNu_MuonIDAndIso = selWMuNu_MuonIDonly.clone()
selWMuNu_MuonIDAndIso.IsoCut03 = cms.untracked.double(0.10) # Put Back Isolation Cut 

selWMuNu_MuonIDAndIsoAndTrigger = selWMuNu_MuonIDAndIso.clone()
selWMuNu_MuonIDAndIsoAndTrigger.MuonTrig = cms.untracked.vstring("HLT_Mu9","HLT_Mu11","HLT_Mu15_v1") #Put Back Trigger 

selWMuNu_MuonSelected = selWMuNu_MuonIDAndIsoAndTrigger.clone() 
selWMuNu_MuonSelected.PtThrForZ1= cms.untracked.double(20.0)
selWMuNu_MuonSelected.PtThrForZ2= cms.untracked.double(10.0)
selWMuNu_MuonSelected.PtCut = cms.untracked.double(25.0)

