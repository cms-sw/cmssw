#import FWCore.ParameterSet.Config as cms

from ElectroWeakAnalysis.WMuNu.wmunusProducer_cfi import *

# Paths for WMuNuSelector filtering of events
# Be careful: events may contain more than one WMunuCandidate if they
#             contain more than one muon. 
# The "real" WMuNuCandidate selected is the first one of the collection

selcorMet = cms.EDFilter("WMuNuSelector",
      # Fill Basc Histograms? ->
      plotHistograms = cms.untracked.bool(False),

      # Input collections ->
      MuonTag = cms.untracked.InputTag("muons"),
      TrigTag = cms.untracked.InputTag("TriggerResults::HLT"),
      JetTag = cms.untracked.InputTag("ak5CaloJets"), # CAREFUL --> If you run on Summer09 MC, this was called "antikt5CaloJets"
      WMuNuCollectionTag = cms.untracked.InputTag("corMetWMuNus"),

      # Preselection! 
      MuonTrig = cms.untracked.string("HLT_L2Mu9"),
      PtThrForZ1 = cms.untracked.double(20.0),
      PtThrForZ2 = cms.untracked.double(10.0),
      EJetMin = cms.untracked.double(40.),
      NJetMax = cms.untracked.int32(999999),

      # Main cuts ->
      PtCut = cms.untracked.double(25.0),
      EtaCut = cms.untracked.double(2.1),
      IsRelativeIso = cms.untracked.bool(True),
      IsCombinedIso = cms.untracked.bool(True), #--> Changed default to Combined Iso. A cut in 0.15 is equivalent (for signal)
      IsoCut03 = cms.untracked.double(0.15),    # to a cut in TrackIso in 0.10
      MtMin = cms.untracked.double(50.0),
      MtMax = cms.untracked.double(200.0),
      MetMin = cms.untracked.double(-999999.),
      MetMax = cms.untracked.double(999999.),
      AcopCut = cms.untracked.double(2.),  # Remember to take this out if you are looking for High-Pt Bosons! (V+Jets)

      # Muon quality cuts ->
      DxyCut = cms.untracked.double(0.2),
      NormalizedChi2Cut = cms.untracked.double(10.),
      TrackerHitsCut = cms.untracked.int32(11),
      MuonHitsCut = cms.untracked.int32(1),
      IsAlsoTrackerMuon = cms.untracked.bool(True),

      # Select only W-, W+ ( default is all Ws)  
      SelectByCharge=cms.untracked.int32(0)

)

selpfMet = cms.EDFilter("WMuNuSelector",
      # Fill Basc Histograms? ->
      plotHistograms = cms.untracked.bool(False),

      # Input collections ->
      MuonTag = cms.untracked.InputTag("muons"),
      TrigTag = cms.untracked.InputTag("TriggerResults::HLT"),
      JetTag = cms.untracked.InputTag("ak5CaloJets"),
      WMuNuCollectionTag = cms.untracked.InputTag("pfMetWMuNus"),

      # Preselection! 
      MuonTrig = cms.untracked.string("HLT_L2Mu9"),
      PtThrForZ1 = cms.untracked.double(20.0),
      PtThrForZ2 = cms.untracked.double(10.0),
      EJetMin = cms.untracked.double(40.),
      NJetMax = cms.untracked.int32(999999),

      # Main cuts -> 
      UseTrackerPt = cms.untracked.bool(True),
      PtCut = cms.untracked.double(25.0),
      EtaCut = cms.untracked.double(2.1),
      IsRelativeIso = cms.untracked.bool(True),
      IsCombinedIso = cms.untracked.bool(True),
      IsoCut03 = cms.untracked.double(0.15),
      MtMin = cms.untracked.double(50.0),
      MtMax = cms.untracked.double(200.0),
      MetMin = cms.untracked.double(-999999.),
      MetMax = cms.untracked.double(999999.),
      AcopCut = cms.untracked.double(2.),

      # Muon quality cuts ->
      DxyCut = cms.untracked.double(0.2),
      NormalizedChi2Cut = cms.untracked.double(10.),
      TrackerHitsCut = cms.untracked.int32(11),
      MuonHitsCut = cms.untracked.int32(1),
      IsAlsoTrackerMuon = cms.untracked.bool(True),

      # Select only W-, W+ ( default is all Ws)
      SelectByCharge=cms.untracked.int32(0)

)

seltcMet = cms.EDFilter("WMuNuSelector",
      # Fill Basc Histograms? ->
      plotHistograms = cms.untracked.bool(False),

      # Input collections ->
      MuonTag = cms.untracked.InputTag("muons"),
      TrigTag = cms.untracked.InputTag("TriggerResults::HLT"),
      JetTag = cms.untracked.InputTag("ak5CaloJets"),
      WMuNuCollectionTag = cms.untracked.InputTag("tcMetWMuNus"),

      # Preselection! 
      MuonTrig = cms.untracked.string("HLT_L2Mu9"),
      PtThrForZ1 = cms.untracked.double(20.0),
      PtThrForZ2 = cms.untracked.double(10.0),
      EJetMin = cms.untracked.double(40.),
      NJetMax = cms.untracked.int32(999999),

      # Main cuts ->
      UseTrackerPt = cms.untracked.bool(True),
      PtCut = cms.untracked.double(25.0),
      EtaCut = cms.untracked.double(2.1),
      IsRelativeIso = cms.untracked.bool(True),
      IsCombinedIso = cms.untracked.bool(True),
      IsoCut03 = cms.untracked.double(0.15),
      MtMin = cms.untracked.double(50.0),
      MtMax = cms.untracked.double(200.0),
      MetMin = cms.untracked.double(-999999.),
      MetMax = cms.untracked.double(999999.),
      AcopCut = cms.untracked.double(2.),

      # Muon quality cuts ->
      DxyCut = cms.untracked.double(0.2),
      NormalizedChi2Cut = cms.untracked.double(10.),
      TrackerHitsCut = cms.untracked.int32(11),
      MuonHitsCut = cms.untracked.int32(1),
      IsAlsoTrackerMuon = cms.untracked.bool(True),

      # Select only W-, W+ ( default is all Ws)
      SelectByCharge=cms.untracked.int32(0)

)

selectCaloMetWMuNus = cms.Sequence(corMetWMuNus+selcorMet)

selectPfMetWMuNus = cms.Sequence(pfMetWMuNus+selpfMet)

selectTcMetWMuNus = cms.Sequence(tcMetWMuNus+seltcMet)


