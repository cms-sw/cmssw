import FWCore.ParameterSet.Config as cms

heavyFlavorValidationHarvesting = cms.EDAnalyzer("HeavyFlavorHarvesting",
  MyDQMrootFolder = cms.untracked.string('HLT/HeavyFlavor/HLT/HLT_Mu5'),
  Efficiencies = cms.untracked.VPSet(
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("globMuon_genEtaPt"),
      DenominatorMEname = cms.untracked.string("genMuon_genEtaPt"),
      EfficiencyMEname = cms.untracked.string("effGlobGen_genEtaPt")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("filt1Muon_recoEtaPt"),
      DenominatorMEname = cms.untracked.string("globMuon_recoEtaPt"),
      EfficiencyMEname = cms.untracked.string("effFilt1Glob_recoEtaPt")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("filt2Muon_recoEtaPt"),
      DenominatorMEname = cms.untracked.string("globMuon_recoEtaPt"),
      EfficiencyMEname = cms.untracked.string("effFilt2Glob_recoEtaPt")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("filt3Muon_recoEtaPt"),
      DenominatorMEname = cms.untracked.string("globMuon_recoEtaPt"),
      EfficiencyMEname = cms.untracked.string("effFilt3Glob_recoEtaPt")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("filt4Muon_recoEtaPt"),
      DenominatorMEname = cms.untracked.string("globMuon_recoEtaPt"),
      EfficiencyMEname = cms.untracked.string("effFilt4Glob_recoEtaPt")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("filt5Muon_recoEtaPt"),
      DenominatorMEname = cms.untracked.string("globMuon_recoEtaPt"),
      EfficiencyMEname = cms.untracked.string("effFilt5Glob_recoEtaPt")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("pathMuon_recoEtaPt"),
      DenominatorMEname = cms.untracked.string("globMuon_recoEtaPt"),
      EfficiencyMEname = cms.untracked.string("effPathGlob_recoEtaPt")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("filt2Muon_recoEtaPt"),
      DenominatorMEname = cms.untracked.string("filt1Muon_recoEtaPt"),
      EfficiencyMEname = cms.untracked.string("effFilt2Filt1_recoEtaPt")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("filt3Muon_recoEtaPt"),
      DenominatorMEname = cms.untracked.string("filt2Muon_recoEtaPt"),
      EfficiencyMEname = cms.untracked.string("effFilt3Filt2_recoEtaPt")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("filt4Muon_recoEtaPt"),
      DenominatorMEname = cms.untracked.string("filt3Muon_recoEtaPt"),
      EfficiencyMEname = cms.untracked.string("effFilt4Filt3_recoEtaPt")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("filt5Muon_recoEtaPt"),
      DenominatorMEname = cms.untracked.string("filt4Muon_recoEtaPt"),
      EfficiencyMEname = cms.untracked.string("effFilt5Filt4_recoEtaPt")
    ),

    
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("globMuon_genEtaPhi"),
      DenominatorMEname = cms.untracked.string("genMuon_genEtaPhi"),
      EfficiencyMEname = cms.untracked.string("effGlobGen_genEtaPhi")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("filt1Muon_recoEtaPhi"),
      DenominatorMEname = cms.untracked.string("globMuon_recoEtaPhi"),
      EfficiencyMEname = cms.untracked.string("effFilt1Glob_recoEtaPhi")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("filt2Muon_recoEtaPhi"),
      DenominatorMEname = cms.untracked.string("globMuon_recoEtaPhi"),
      EfficiencyMEname = cms.untracked.string("effFilt2Glob_recoEtaPhi")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("filt3Muon_recoEtaPhi"),
      DenominatorMEname = cms.untracked.string("globMuon_recoEtaPhi"),
      EfficiencyMEname = cms.untracked.string("effFilt3Glob_recoEtaPhi")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("filt4Muon_recoEtaPhi"),
      DenominatorMEname = cms.untracked.string("globMuon_recoEtaPhi"),
      EfficiencyMEname = cms.untracked.string("effFilt4Glob_recoEtaPhi")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("filt5Muon_recoEtaPhi"),
      DenominatorMEname = cms.untracked.string("globMuon_recoEtaPhi"),
      EfficiencyMEname = cms.untracked.string("effFilt5Glob_recoEtaPhi")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("pathMuon_recoEtaPhi"),
      DenominatorMEname = cms.untracked.string("globMuon_recoEtaPhi"),
      EfficiencyMEname = cms.untracked.string("effPathGlob_recoEtaPhi")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("filt2Muon_recoEtaPhi"),
      DenominatorMEname = cms.untracked.string("filt1Muon_recoEtaPhi"),
      EfficiencyMEname = cms.untracked.string("effFilt2Filt1_recoEtaPhi")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("filt3Muon_recoEtaPhi"),
      DenominatorMEname = cms.untracked.string("filt2Muon_recoEtaPhi"),
      EfficiencyMEname = cms.untracked.string("effFilt3Filt2_recoEtaPhi")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("filt4Muon_recoEtaPhi"),
      DenominatorMEname = cms.untracked.string("filt3Muon_recoEtaPhi"),
      EfficiencyMEname = cms.untracked.string("effFilt4Filt3_recoEtaPhi")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("filt5Muon_recoEtaPhi"),
      DenominatorMEname = cms.untracked.string("filt4Muon_recoEtaPhi"),
      EfficiencyMEname = cms.untracked.string("effFilt5Filt4_recoEtaPhi")
    ),

#################  DOUBLE  ETA  PT  ######################
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("globDimuon_genEtaPt"),
      DenominatorMEname = cms.untracked.string("genDimuon_genEtaPt"),
      EfficiencyMEname = cms.untracked.string("effGlobDigenAND_genEtaPt")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("filt1Dimuon_recoEtaPt"),
      DenominatorMEname = cms.untracked.string("globDimuon_recoEtaPt"),
      EfficiencyMEname = cms.untracked.string("effFilt1DiglobOR_recoEtaPt")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("filt2Dimuon_recoEtaPt"),
      DenominatorMEname = cms.untracked.string("globDimuon_recoEtaPt"),
      EfficiencyMEname = cms.untracked.string("effFilt2DiglobOR_recoEtaPt")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("filt3Dimuon_recoEtaPt"),
      DenominatorMEname = cms.untracked.string("globDimuon_recoEtaPt"),
      EfficiencyMEname = cms.untracked.string("effFilt3DiglobOR_recoEtaPt")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("filt4Dimuon_recoEtaPt"),
      DenominatorMEname = cms.untracked.string("globDimuon_recoEtaPt"),
      EfficiencyMEname = cms.untracked.string("effFilt4DiglobOR_recoEtaPt")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("filt5Dimuon_recoEtaPt"),
      DenominatorMEname = cms.untracked.string("globDimuon_recoEtaPt"),
      EfficiencyMEname = cms.untracked.string("effFilt5DiglobOR_recoEtaPt")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("pathDimuon_recoEtaPt"),
      DenominatorMEname = cms.untracked.string("globDimuon_recoEtaPt"),
      EfficiencyMEname = cms.untracked.string("effPathDiglobOR_recoEtaPt")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("diFilt1Dimuon_recoEtaPt"),
      DenominatorMEname = cms.untracked.string("globDimuon_recoEtaPt"),
      EfficiencyMEname = cms.untracked.string("effFilt1DiglobAND_recoEtaPt")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("diFilt2Dimuon_recoEtaPt"),
      DenominatorMEname = cms.untracked.string("globDimuon_recoEtaPt"),
      EfficiencyMEname = cms.untracked.string("effFilt2DiglobAND_recoEtaPt")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("diFilt3Dimuon_recoEtaPt"),
      DenominatorMEname = cms.untracked.string("globDimuon_recoEtaPt"),
      EfficiencyMEname = cms.untracked.string("effFilt3DiglobAND_recoEtaPt")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("diFilt4Dimuon_recoEtaPt"),
      DenominatorMEname = cms.untracked.string("globDimuon_recoEtaPt"),
      EfficiencyMEname = cms.untracked.string("effFilt4DiglobAND_recoEtaPt")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("diFilt5Dimuon_recoEtaPt"),
      DenominatorMEname = cms.untracked.string("globDimuon_recoEtaPt"),
      EfficiencyMEname = cms.untracked.string("effFilt5DiglobAND_recoEtaPt")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("diPathDimuon_recoEtaPt"),
      DenominatorMEname = cms.untracked.string("globDimuon_recoEtaPt"),
      EfficiencyMEname = cms.untracked.string("effPathDiglobAND_recoEtaPt")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("diFilt2Dimuon_recoEtaPt"),
      DenominatorMEname = cms.untracked.string("diFilt1Dimuon_recoEtaPt"),
      EfficiencyMEname = cms.untracked.string("effFilt2Difilt1AND_recoEtaPt")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("diFilt3Dimuon_recoEtaPt"),
      DenominatorMEname = cms.untracked.string("diFilt2Dimuon_recoEtaPt"),
      EfficiencyMEname = cms.untracked.string("effFilt3Difilt2AND_recoEtaPt")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("diFilt4Dimuon_recoEtaPt"),
      DenominatorMEname = cms.untracked.string("diFilt3Dimuon_recoEtaPt"),
      EfficiencyMEname = cms.untracked.string("effFilt4Difilt3AND_recoEtaPt")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("diFilt5Dimuon_recoEtaPt"),
      DenominatorMEname = cms.untracked.string("diFilt4Dimuon_recoEtaPt"),
      EfficiencyMEname = cms.untracked.string("effFilt5Difilt4AND_recoEtaPt")
    ),

    
#################  DOUBLE  RAPIDITY  PT  ######################
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("globDimuon_genRapPt"),
      DenominatorMEname = cms.untracked.string("genDimuon_genRapPt"),
      EfficiencyMEname = cms.untracked.string("effGlobDigenAND_genRapPt")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("filt1Dimuon_recoRapPt"),
      DenominatorMEname = cms.untracked.string("globDimuon_recoRapPt"),
      EfficiencyMEname = cms.untracked.string("effFilt1DiglobOR_recoRapPt")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("filt2Dimuon_recoRapPt"),
      DenominatorMEname = cms.untracked.string("globDimuon_recoRapPt"),
      EfficiencyMEname = cms.untracked.string("effFilt2DiglobOR_recoRapPt")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("filt3Dimuon_recoRapPt"),
      DenominatorMEname = cms.untracked.string("globDimuon_recoRapPt"),
      EfficiencyMEname = cms.untracked.string("effFilt3DiglobOR_recoRapPt")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("filt4Dimuon_recoRapPt"),
      DenominatorMEname = cms.untracked.string("globDimuon_recoRapPt"),
      EfficiencyMEname = cms.untracked.string("effFilt4DiglobOR_recoRapPt")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("filt5Dimuon_recoRapPt"),
      DenominatorMEname = cms.untracked.string("globDimuon_recoRapPt"),
      EfficiencyMEname = cms.untracked.string("effFilt5DiglobOR_recoRapPt")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("pathDimuon_recoRapPt"),
      DenominatorMEname = cms.untracked.string("globDimuon_recoRapPt"),
      EfficiencyMEname = cms.untracked.string("effPathDiglobOR_recoRapPt")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("diFilt1Dimuon_recoRapPt"),
      DenominatorMEname = cms.untracked.string("globDimuon_recoRapPt"),
      EfficiencyMEname = cms.untracked.string("effFilt1DiglobAND_recoRapPt")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("diFilt2Dimuon_recoRapPt"),
      DenominatorMEname = cms.untracked.string("globDimuon_recoRapPt"),
      EfficiencyMEname = cms.untracked.string("effFilt2DiglobAND_recoRapPt")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("diFilt3Dimuon_recoRapPt"),
      DenominatorMEname = cms.untracked.string("globDimuon_recoRapPt"),
      EfficiencyMEname = cms.untracked.string("effFilt3DiglobAND_recoRapPt")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("diFilt4Dimuon_recoRapPt"),
      DenominatorMEname = cms.untracked.string("globDimuon_recoRapPt"),
      EfficiencyMEname = cms.untracked.string("effFilt4DiglobAND_recoRapPt")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("diFilt5Dimuon_recoRapPt"),
      DenominatorMEname = cms.untracked.string("globDimuon_recoRapPt"),
      EfficiencyMEname = cms.untracked.string("effFilt5DiglobAND_recoRapPt")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("diPathDimuon_recoRapPt"),
      DenominatorMEname = cms.untracked.string("globDimuon_recoRapPt"),
      EfficiencyMEname = cms.untracked.string("effPathDiglobAND_recoRapPt")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("diFilt2Dimuon_recoRapPt"),
      DenominatorMEname = cms.untracked.string("diFilt1Dimuon_recoRapPt"),
      EfficiencyMEname = cms.untracked.string("effFilt2Difilt1AND_recoRapPt")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("diFilt3Dimuon_recoRapPt"),
      DenominatorMEname = cms.untracked.string("diFilt2Dimuon_recoRapPt"),
      EfficiencyMEname = cms.untracked.string("effFilt3Difilt2AND_recoRapPt")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("diFilt4Dimuon_recoRapPt"),
      DenominatorMEname = cms.untracked.string("diFilt3Dimuon_recoRapPt"),
      EfficiencyMEname = cms.untracked.string("effFilt4Difilt3AND_recoRapPt")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("diFilt5Dimuon_recoRapPt"),
      DenominatorMEname = cms.untracked.string("diFilt4Dimuon_recoRapPt"),
      EfficiencyMEname = cms.untracked.string("effFilt5Difilt4AND_recoRapPt")
    ),

    

    
    #################  DOUBLE  PT  DR  ######################
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("globDimuon_genPtDR"),
      DenominatorMEname = cms.untracked.string("genDimuon_genPtDR"),
      EfficiencyMEname = cms.untracked.string("effGlobDigenAND_genPtDR")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("filt1Dimuon_recoPtDR"),
      DenominatorMEname = cms.untracked.string("globDimuon_recoPtDR"),
      EfficiencyMEname = cms.untracked.string("effFilt1DiglobOR_recoPtDR")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("filt2Dimuon_recoPtDR"),
      DenominatorMEname = cms.untracked.string("globDimuon_recoPtDR"),
      EfficiencyMEname = cms.untracked.string("effFilt2DiglobOR_recoPtDR")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("filt3Dimuon_recoPtDR"),
      DenominatorMEname = cms.untracked.string("globDimuon_recoPtDR"),
      EfficiencyMEname = cms.untracked.string("effFilt3DiglobOR_recoPtDR")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("filt4Dimuon_recoPtDR"),
      DenominatorMEname = cms.untracked.string("globDimuon_recoPtDR"),
      EfficiencyMEname = cms.untracked.string("effFilt4DiglobOR_recoPtDR")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("filt5Dimuon_recoPtDR"),
      DenominatorMEname = cms.untracked.string("globDimuon_recoPtDR"),
      EfficiencyMEname = cms.untracked.string("effFilt5DiglobOR_recoPtDR")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("pathDimuon_recoPtDR"),
      DenominatorMEname = cms.untracked.string("globDimuon_recoPtDR"),
      EfficiencyMEname = cms.untracked.string("effPathDiglobOR_recoPtDR")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("diFilt1Dimuon_recoPtDR"),
      DenominatorMEname = cms.untracked.string("globDimuon_recoPtDR"),
      EfficiencyMEname = cms.untracked.string("effFilt1DiglobAND_recoPtDR")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("diFilt2Dimuon_recoPtDR"),
      DenominatorMEname = cms.untracked.string("globDimuon_recoPtDR"),
      EfficiencyMEname = cms.untracked.string("effFilt2DiglobAND_recoPtDR")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("diFilt3Dimuon_recoPtDR"),
      DenominatorMEname = cms.untracked.string("globDimuon_recoPtDR"),
      EfficiencyMEname = cms.untracked.string("effFilt3DiglobAND_recoPtDR")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("diFilt4Dimuon_recoPtDR"),
      DenominatorMEname = cms.untracked.string("globDimuon_recoPtDR"),
      EfficiencyMEname = cms.untracked.string("effFilt4DiglobAND_recoPtDR")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("diFilt5Dimuon_recoPtDR"),
      DenominatorMEname = cms.untracked.string("globDimuon_recoPtDR"),
      EfficiencyMEname = cms.untracked.string("effFilt5DiglobAND_recoPtDR")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("diPathDimuon_recoPtDR"),
      DenominatorMEname = cms.untracked.string("globDimuon_recoPtDR"),
      EfficiencyMEname = cms.untracked.string("effPathDiglobAND_recoPtDR")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("diFilt2Dimuon_recoPtDR"),
      DenominatorMEname = cms.untracked.string("diFilt1Dimuon_recoPtDR"),
      EfficiencyMEname = cms.untracked.string("effFilt2Difilt1AND_recoPtDR")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("diFilt3Dimuon_recoPtDR"),
      DenominatorMEname = cms.untracked.string("diFilt2Dimuon_recoPtDR"),
      EfficiencyMEname = cms.untracked.string("effFilt3Difilt2AND_recoPtDR")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("diFilt4Dimuon_recoPtDR"),
      DenominatorMEname = cms.untracked.string("diFilt3Dimuon_recoPtDR"),
      EfficiencyMEname = cms.untracked.string("effFilt4Difilt3AND_recoPtDR")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("diFilt5Dimuon_recoPtDR"),
      DenominatorMEname = cms.untracked.string("diFilt4Dimuon_recoPtDR"),
      EfficiencyMEname = cms.untracked.string("effFilt5Difilt4AND_recoPtDR")
    ),

    

    #################  DOUBLE  ETA  PT  ######################
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("globDimuon_genPtDRpos"),
      DenominatorMEname = cms.untracked.string("genDimuon_genPtDRpos"),
      EfficiencyMEname = cms.untracked.string("effGlobDigenAND_genPtDRpos")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("filt1Dimuon_recoPtDRpos"),
      DenominatorMEname = cms.untracked.string("globDimuon_recoPtDRpos"),
      EfficiencyMEname = cms.untracked.string("effFilt1DiglobOR_recoPtDRpos")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("filt2Dimuon_recoPtDRpos"),
      DenominatorMEname = cms.untracked.string("globDimuon_recoPtDRpos"),
      EfficiencyMEname = cms.untracked.string("effFilt2DiglobOR_recoPtDRpos")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("filt3Dimuon_recoPtDRpos"),
      DenominatorMEname = cms.untracked.string("globDimuon_recoPtDRpos"),
      EfficiencyMEname = cms.untracked.string("effFilt3DiglobOR_recoPtDRpos")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("filt4Dimuon_recoPtDRpos"),
      DenominatorMEname = cms.untracked.string("globDimuon_recoPtDRpos"),
      EfficiencyMEname = cms.untracked.string("effFilt4DiglobOR_recoPtDRpos")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("filt5Dimuon_recoPtDRpos"),
      DenominatorMEname = cms.untracked.string("globDimuon_recoPtDRpos"),
      EfficiencyMEname = cms.untracked.string("effFilt5DiglobOR_recoPtDRpos")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("pathDimuon_recoPtDRpos"),
      DenominatorMEname = cms.untracked.string("globDimuon_recoPtDRpos"),
      EfficiencyMEname = cms.untracked.string("effPathDiglobOR_recoPtDRpos")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("diFilt1Dimuon_recoPtDRpos"),
      DenominatorMEname = cms.untracked.string("globDimuon_recoPtDRpos"),
      EfficiencyMEname = cms.untracked.string("effFilt1DiglobAND_recoPtDRpos")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("diFilt2Dimuon_recoPtDRpos"),
      DenominatorMEname = cms.untracked.string("globDimuon_recoPtDRpos"),
      EfficiencyMEname = cms.untracked.string("effFilt2DiglobAND_recoPtDRpos")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("diFilt3Dimuon_recoPtDRpos"),
      DenominatorMEname = cms.untracked.string("globDimuon_recoPtDRpos"),
      EfficiencyMEname = cms.untracked.string("effFilt3DiglobAND_recoPtDRpos")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("diFilt4Dimuon_recoPtDRpos"),
      DenominatorMEname = cms.untracked.string("globDimuon_recoPtDRpos"),
      EfficiencyMEname = cms.untracked.string("effFilt4DiglobAND_recoPtDRpos")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("diFilt5Dimuon_recoPtDRpos"),
      DenominatorMEname = cms.untracked.string("globDimuon_recoPtDRpos"),
      EfficiencyMEname = cms.untracked.string("effFilt5DiglobAND_recoPtDRpos")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("diPathDimuon_recoPtDRpos"),
      DenominatorMEname = cms.untracked.string("globDimuon_recoPtDRpos"),
      EfficiencyMEname = cms.untracked.string("effPathDiglobAND_recoPtDRpos")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("diFilt2Dimuon_recoPtDRpos"),
      DenominatorMEname = cms.untracked.string("diFilt1Dimuon_recoPtDRpos"),
      EfficiencyMEname = cms.untracked.string("effFilt2Difilt1AND_recoPtDRpos")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("diFilt3Dimuon_recoPtDRpos"),
      DenominatorMEname = cms.untracked.string("diFilt2Dimuon_recoPtDRpos"),
      EfficiencyMEname = cms.untracked.string("effFilt3Difilt2AND_recoPtDRpos")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("diFilt4Dimuon_recoPtDRpos"),
      DenominatorMEname = cms.untracked.string("diFilt3Dimuon_recoPtDRpos"),
      EfficiencyMEname = cms.untracked.string("effFilt4Difilt3AND_recoPtDRpos")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("diFilt5Dimuon_recoPtDRpos"),
      DenominatorMEname = cms.untracked.string("diFilt4Dimuon_recoPtDRpos"),
      EfficiencyMEname = cms.untracked.string("effFilt5Difilt4AND_recoPtDRpos")
    ),

    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("resultDimuon_recoEtaPt"),
      DenominatorMEname = cms.untracked.string("globDimuon_recoEtaPt"),
      EfficiencyMEname = cms.untracked.string("effResultDiglob_recoEtaPt")
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("resultDimuon_recoRapPt"),
      DenominatorMEname = cms.untracked.string("globDimuon_recoRapPt"),
      EfficiencyMEname = cms.untracked.string("effResultDiglob_recoRapPt")
    )

  )
)
