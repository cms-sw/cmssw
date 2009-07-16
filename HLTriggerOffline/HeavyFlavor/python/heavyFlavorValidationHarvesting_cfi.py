import FWCore.ParameterSet.Config as cms

heavyFlavorValidationHarvesting = cms.EDAnalyzer("DQMGenericClient",
  subDirs        = cms.untracked.vstring('HLT/HeavyFlavor/*'),
  verbose        = cms.untracked.uint32(0),
#  outputFileName = cms.untracked.string('heavyFlavorValidationHarvesting.root'),
  commands       = cms.vstring(),
  resolution     = cms.vstring(),                                    
  efficiency     = cms.vstring(
    "HLT_DoubleMu0/eff_genGlobDimuonPath 'HLT_DoubleMu0 / Global Dimuon;Global Dimuon p_{T} (GeV);Efficiency' HLT_DoubleMu0/genGlobDimuonPath_recoPt OfflineMuons/genGlobDimuon_recoPt",
    "HLT_DoubleMu0/eff_genGlobL1Dimuon 'HLT_DoubleMu0 (L1) / Global Dimuon;Global Dimuon p_{T} (GeV);Efficiency' HLT_DoubleMu0/genGlobL1Dimuon_recoPt OfflineMuons/genGlobDimuon_recoPt",
    "HLT_DoubleMu0/eff_genGlobL1L2L2vDimuon 'HLT_DoubleMu0 (L2) / Global Dimuon;Global Dimuon p_{T} (GeV);Efficiency' HLT_DoubleMu0/genGlobL1L2L2vDimuon_recoPt HLT_DoubleMu0/genGlobL1Dimuon_recoPt",
    "HLT_DoubleMu0/eff_genGlobL1L2L2vL3Dimuon 'HLT_DoubleMu0 (L3) / Global Dimuon;Global Dimuon p_{T} (GeV);Efficiency' HLT_DoubleMu0/genGlobL1L2L2vL3Dimuon_recoPt HLT_DoubleMu0/genGlobL1L2L2vDimuon_recoPt",

    "HLT_DoubleMu3/eff_genGlobDimuonPath 'HLT_DoubleMu3 / Global Dimuon;Global Dimuon p_{T} (GeV);Efficiency' HLT_DoubleMu3/genGlobDimuonPath_recoPt OfflineMuons/genGlobDimuon_recoPt",
    "HLT_DoubleMu3/eff_genGlobL1Dimuon 'HLT_DoubleMu3 (L1) / Global Dimuon;Global Dimuon p_{T} (GeV);Efficiency' HLT_DoubleMu3/genGlobL1Dimuon_recoPt OfflineMuons/genGlobDimuon_recoPt",
    "HLT_DoubleMu3/eff_genGlobL1L2L2vDimuon 'HLT_DoubleMu3 (L2) / Global Dimuon;Global Dimuon p_{T} (GeV);Efficiency' HLT_DoubleMu3/genGlobL1L2L2vDimuon_recoPt HLT_DoubleMu3/genGlobL1Dimuon_recoPt",
    "HLT_DoubleMu3/eff_genGlobL1L2L2vL3Dimuon 'HLT_DoubleMu3 (L3) / Global Dimuon;Global Dimuon p_{T} (GeV);Efficiency' HLT_DoubleMu3/genGlobL1L2L2vL3Dimuon_recoPt HLT_DoubleMu3/genGlobL1L2L2vDimuon_recoPt",

    "HLT_L1DoubleMuOpen/eff_genGlobDimuonPath 'HLT_L1DoubleMuOpen / Global Dimuon;Global Dimuon p_{T} (GeV);Efficiency' HLT_L1DoubleMuOpen/genGlobDimuonPath_recoPt OfflineMuons/genGlobDimuon_recoPt",
    "HLT_L1DoubleMuOpen/eff_genGlobL1Dimuon 'HLT_L1DoubleMuOpen (L1) / Global Dimuon;Global Dimuon p_{T} (GeV);Efficiency' HLT_L1DoubleMuOpen/genGlobL1Dimuon_recoPt OfflineMuons/genGlobDimuon_recoPt",
    "HLT_L1DoubleMuOpen/eff_genGlobL1L2L2vDimuon 'HLT_L1DoubleMuOpen (L2) / Global Dimuon;Global Dimuon p_{T} (GeV);Efficiency' HLT_L1DoubleMuOpen/genGlobL1L2L2vDimuon_recoPt HLT_L1DoubleMuOpen/genGlobL1Dimuon_recoPt",
    "HLT_L1DoubleMuOpen/eff_genGlobL1L2L2vL3Dimuon 'HLT_L1DoubleMuOpen (L3) / Global Dimuon;Global Dimuon p_{T} (GeV);Efficiency' HLT_L1DoubleMuOpen/genGlobL1L2L2vL3Dimuon_recoPt HLT_L1DoubleMuOpen/genGlobL1L2L2vDimuon_recoPt",

    "HLT_L1Mu/eff_genGlobDimuonPath 'HLT_L1Mu / Global Dimuon;Global Dimuon p_{T} (GeV);Efficiency' HLT_L1Mu/genGlobDimuonPath_recoPt OfflineMuons/genGlobDimuon_recoPt",
    "HLT_L1Mu/eff_genGlobL1Dimuon 'HLT_L1Mu (L1) / Global Dimuon;Global Dimuon p_{T} (GeV);Efficiency' HLT_L1Mu/genGlobL1Dimuon_recoPt OfflineMuons/genGlobDimuon_recoPt",
    "HLT_L1Mu/eff_genGlobL1L2L2vDimuon 'HLT_L1Mu (L2) / Global Dimuon;Global Dimuon p_{T} (GeV);Efficiency' HLT_L1Mu/genGlobL1L2L2vDimuon_recoPt HLT_L1Mu/genGlobL1Dimuon_recoPt",
    "HLT_L1Mu/eff_genGlobL1L2L2vL3Dimuon 'HLT_L1Mu (L3) / Global Dimuon;Global Dimuon p_{T} (GeV);Efficiency' HLT_L1Mu/genGlobL1L2L2vL3Dimuon_recoPt HLT_L1Mu/genGlobL1L2L2vDimuon_recoPt",

    "HLT_L1MuOpen/eff_genGlobDimuonPath 'HLT_L1MuOpen / Global Dimuon;Global Dimuon p_{T} (GeV);Efficiency' HLT_L1MuOpen/genGlobDimuonPath_recoPt OfflineMuons/genGlobDimuon_recoPt",
    "HLT_L1MuOpen/eff_genGlobL1Dimuon 'HLT_L1MuOpen (L1) / Global Dimuon;Global Dimuon p_{T} (GeV);Efficiency' HLT_L1MuOpen/genGlobL1Dimuon_recoPt OfflineMuons/genGlobDimuon_recoPt",
    "HLT_L1MuOpen/eff_genGlobL1L2L2vDimuon 'HLT_L1MuOpen (L2) / Global Dimuon;Global Dimuon p_{T} (GeV);Efficiency' HLT_L1MuOpen/genGlobL1L2L2vDimuon_recoPt HLT_L1MuOpen/genGlobL1Dimuon_recoPt",
    "HLT_L1MuOpen/eff_genGlobL1L2L2vL3Dimuon 'HLT_L1MuOpen (L3) / Global Dimuon;Global Dimuon p_{T} (GeV);Efficiency' HLT_L1MuOpen/genGlobL1L2L2vL3Dimuon_recoPt HLT_L1MuOpen/genGlobL1L2L2vDimuon_recoPt",

    "HLT_Mu3/eff_genGlobDimuonPath 'HLT_Mu3 / Global Dimuon;Global Dimuon p_{T} (GeV);Efficiency' HLT_Mu3/genGlobDimuonPath_recoPt OfflineMuons/genGlobDimuon_recoPt",
    "HLT_Mu3/eff_genGlobL1Dimuon 'HLT_Mu3 (L1) / Global Dimuon;Global Dimuon p_{T} (GeV);Efficiency' HLT_Mu3/genGlobL1Dimuon_recoPt OfflineMuons/genGlobDimuon_recoPt",
    "HLT_Mu3/eff_genGlobL1L2L2vDimuon 'HLT_Mu3 (L2) / Global Dimuon;Global Dimuon p_{T} (GeV);Efficiency' HLT_Mu3/genGlobL1L2L2vDimuon_recoPt HLT_Mu3/genGlobL1Dimuon_recoPt",
    "HLT_Mu3/eff_genGlobL1L2L2vL3Dimuon 'HLT_Mu3 (L3) / Global Dimuon;Global Dimuon p_{T} (GeV);Efficiency' HLT_Mu3/genGlobL1L2L2vL3Dimuon_recoPt HLT_Mu3/genGlobL1L2L2vDimuon_recoPt",

    "OfflineMuons/eff_genGlobDimuon 'Global Dimuon / Generated Dimuon;Generated Dimuon p_{T} (GeV);Efficiency' OfflineMuons/genGlobDimuon_genPt OfflineMuons/genDimuon_genPt",

    "dR/eff_genGlobDimuon 'Global Dimuon / Generated Dimuon;Generated Dimuon dR At IP;Efficiency' dR/genGlobDimuon_dR dR/genDimuon_dR",
    "dR/eff_genGlobL1Dimuon 'L1 Dimuon / Global Dimuon;Global Dimuon dR At IP;Efficiency' dR/genGlobL1Dimuon_dR dR/genGlobDimuon_dR",
    "dR/eff_genGlobL1L2L2vDimuon 'L2 Dimuon / Global Dimuon;Global Dimuon dR At IP;Efficiency' dR/genGlobL1L2L2vDimuon_dR dR/genGlobL1Dimuon_dR",
    "dR/eff_genGlobL1L2L2vL3Dimuon 'L3 Dimuon / Global Dimuon;Global Dimuon dR At IP;Efficiency' dR/genGlobL1L2L2vL3Dimuon_dR dR/genGlobL1L2L2vDimuon_dR",

    "dR/eff_genGlobL1Dimuon_pos 'L1 Dimuon / Global Dimuon;Global Dimuon dR in Muon System;Efficiency' dR/genGlobL1Dimuon_dRpos dR/genGlobDimuon_dRpos",
    "dR/eff_genGlobL1L2L2vDimuon_pos 'L2 Dimuon / Global Dimuon;Global Dimuon dR in Muon System;Efficiency' dR/genGlobL1L2L2vDimuon_dRpos dR/genGlobL1Dimuon_dRpos",
    "dR/eff_genGlobL1L2L2vL3Dimuon_pos 'L3 Dimuon / Global Dimuon;Global Dimuon dR in Muon System;Efficiency' dR/genGlobL1L2L2vL3Dimuon_dRpos dR/genGlobL1L2L2vDimuon_dRpos"
  )
)
