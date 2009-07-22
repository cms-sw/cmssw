import FWCore.ParameterSet.Config as cms

heavyFlavorValidationHarvesting = cms.EDAnalyzer("DQMGenericClient",
  subDirs        = cms.untracked.vstring('HLT/HeavyFlavor/*'),
  verbose        = cms.untracked.uint32(0),
#  outputFileName = cms.untracked.string('heavyFlavorValidationHarvesting.root'),
  commands       = cms.vstring(),
  resolution     = cms.vstring(),                                    
  efficiency     = cms.vstring(
    "eff_HLT_Mu3 'HLT_Mu3 / Global Quarkonium;Global Quarkonium p_{T} (GeV);Efficiency' HLT_Mu3 denominator_genGlobQuarkonium_recoPt",
    "eff_HLT_Mu5 'HLT_Mu5 / Global Quarkonium;Global Quarkonium p_{T} (GeV);Efficiency' HLT_Mu5 denominator_genGlobQuarkonium_recoPt",
    "eff_HLT_Mu7 'HLT_Mu7 / Global Quarkonium;Global Quarkonium p_{T} (GeV);Efficiency' HLT_Mu7 denominator_genGlobQuarkonium_recoPt",
    "eff_HLT_Mu9 'HLT_Mu9 / Global Quarkonium;Global Quarkonium p_{T} (GeV);Efficiency' HLT_Mu9 denominator_genGlobQuarkonium_recoPt",
    "eff_HLT_DoubleMu3 'HLT_DoubleMu3 / Global Quarkonium;Global Quarkonium p_{T} (GeV);Efficiency' HLT_DoubleMu3 denominator_genGlobQuarkonium_recoPt",
    "eff_HLT_DoubleMu3_JPsi 'HLT_DoubleMu3_JPsi / Global Quarkonium;Global Quarkonium p_{T} (GeV);Efficiency' HLT_DoubleMu3_JPsi denominator_genGlobQuarkonium_recoPt",
    "eff_HLT_DoubleMu3_Upsilon 'HLT_DoubleMu3_Upsilon / Global Quarkonium;Global Quarkonium p_{T} (GeV);Efficiency' HLT_DoubleMu3_Upsilon denominator_genGlobQuarkonium_recoPt",
    "eff_HLT_DoubleMu3_SameSign 'HLT_DoubleMu3_SameSign / Global Quarkonium;Global Quarkonium p_{T} (GeV);Efficiency' HLT_DoubleMu3_SameSign denominator_genGlobQuarkonium_recoPt",
    "eff_HLT_DoubleMu3_BJPsi 'HLT_DoubleMu3_BJPsi / Global Quarkonium;Global Quarkonium p_{T} (GeV);Efficiency' HLT_DoubleMu3_BJPsi denominator_genGlobQuarkonium_recoPt",
    "eff_HLT_DoubleMu3_Vtx2mm 'HLT_DoubleMu3_Vtx2mm / Global Quarkonium;Global Quarkonium p_{T} (GeV);Efficiency' HLT_DoubleMu3_Vtx2mm denominator_genGlobQuarkonium_recoPt",
    "eff_HLT_DoubleIsoMu3 'HLT_DoubleIsoMu3 / Global Quarkonium;Global Quarkonium p_{T} (GeV);Efficiency' HLT_DoubleIsoMu3 denominator_genGlobQuarkonium_recoPt",
    "eff_HLT_L1Mu 'HLT_L1Mu / Global Quarkonium;Global Quarkonium p_{T} (GeV);Efficiency' HLT_L1Mu denominator_genGlobQuarkonium_recoPt",
    "eff_HLT_L1MuOpen 'HLT_L1MuOpen / Global Quarkonium;Global Quarkonium p_{T} (GeV);Efficiency' HLT_L1MuOpen denominator_genGlobQuarkonium_recoPt"
  )
)
