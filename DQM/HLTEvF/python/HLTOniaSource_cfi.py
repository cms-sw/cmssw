import FWCore.ParameterSet.Config as cms

hltOniaSource = cms.EDAnalyzer("HLTOniaSource",
      # DQM Folder
      SubSystemFolder = cms.untracked.string("HLT/HLTMonMuon/Onia"),
      # Trigger Process info
      TriggerProcessName = cms.untracked.string(""),  # --- Fill only for DEBUG reasons ---
      TriggerPathNames = cms.untracked.vstring("HLT_Mu0_Track0_Jpsi","HLT_Mu3_Track0_Jpsi","HLT_Mu5_Track0_Jpsi"),# --- Fill only for DEBUG reasons ---
      TriggerSummaryTag  = cms.untracked.InputTag("hltTriggerSummaryRAW", "", "HLT"),
      # Tags for Onia filters
      OniaMuonTag = cms.untracked.VInputTag(cms.untracked.InputTag("hltMu0TrackJpsiL3Filtered0", "", "HLT"), cms.untracked.InputTag("hltMu3TrackJpsiL3Filtered3", "", "HLT"), cms.untracked.InputTag("hltMu5TrackJpsiL3Filtered5", "", "HLT")),
                               
      PixelTagAfterFilter = cms.untracked.VInputTag(cms.untracked.InputTag("hltMu0TrackJpsiPixelMassFiltered", "", "HLT"),cms.untracked.InputTag("hltMu3TrackJpsiPixelMassFiltered", "", "HLT"), cms.untracked.InputTag("hltMu5TrackJpsiPixelMassFiltered", "", "HLT")),
                               
      TrackTagAfterFilter =cms.untracked.VInputTag(cms.untracked.InputTag("hltMu0TrackJpsiTrackMassFiltered", "", "HLT"),cms.untracked.InputTag("hltMu3TrackJpsiTrackMassFiltered", "", "HLT"), cms.untracked.InputTag("hltMu5TrackJpsiTrackMassFiltered", "", "HLT")),
      #Pixel Tag BEFORE filters
      PixelTag = cms.untracked.InputTag("hltPixelTracks", "", "HLT"),
      #Tracker Track Tag BEFORE filters
      TrackTag = cms.untracked.InputTag("hltMuTrackJpsiCtfTrackCands", "", "HLT"),
      #Beam Spot Tag
      BeamSpotTag = cms.untracked.InputTag("hltOfflineBeamSpot", "", "HLT")
      )                
      
