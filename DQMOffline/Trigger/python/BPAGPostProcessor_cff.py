import FWCore.ParameterSet.Config as cms

JpsiPars = cms.untracked.PSet(
      MassDimension = cms.untracked.int32(2),
      FitFunction = cms.untracked.string("GaussianPlusLinear"),
      ExpectedMean = cms.untracked.double(3.1),
      ExpectedSigma = cms.untracked.double(0.005),
      Width = cms.untracked.double(0),
      FitRangeLow = cms.untracked.double(2.6),
      FitRangeHigh = cms.untracked.double(3.6),
      SignalRangeLow = cms.untracked.double(2.9),
      SignalRangeHigh = cms.untracked.double(3.3)
)

JpsiClient = cms.EDAnalyzer("DQMGenericTnPClient",
  MyDQMrootFolder = cms.untracked.string("HLT/Muon/Distributions"),
  # Set this if you want to save info about each fit
  #SavePlotsInRootFileName = cms.untracked.string("fittingPlots.root"),
  SavePlotsInRootFileName = cms.untracked.string(""),						 
  Verbose = cms.untracked.bool(False),
  Efficiencies = cms.untracked.VPSet(
    JpsiPars.clone(
      NumeratorMEname = cms.untracked.string("HLT_Mu3/probeAnyMuon/diMuonMassVsPt_L3Filtered"),
      DenominatorMEname = cms.untracked.string("HLT_Mu3/probeAnyMuon/diMuonMassVsPt_All"),
      EfficiencyMEname = cms.untracked.string("HLT_Mu3/probeAnyMuon/effVsPt"),
    ),
    JpsiPars.clone(
      NumeratorMEname = cms.untracked.string("HLT_Mu3/probeAnyMuon/diMuonMassVsEta_L3Filtered"),
      DenominatorMEname = cms.untracked.string("HLT_Mu3/probeAnyMuon/diMuonMassVsEta_All"),
      EfficiencyMEname = cms.untracked.string("HLT_Mu3/probeAnyMuon/effVsEta"),
    ),
    JpsiPars.clone(
      NumeratorMEname = cms.untracked.string("HLT_Mu3/probeAnyMuon/diMuonMassVsPhi_L3Filtered"),
      DenominatorMEname = cms.untracked.string("HLT_Mu3/probeAnyMuon/diMuonMassVsPhi_All"),
      EfficiencyMEname = cms.untracked.string("HLT_Mu3/probeAnyMuon/effVsPhi"),
    ),
    JpsiPars.clone(
      NumeratorMEname = cms.untracked.string("HLT_Mu5/probeAnyMuon/diMuonMassVsPt_L3Filtered"),
      DenominatorMEname = cms.untracked.string("HLT_Mu5/probeAnyMuon/diMuonMassVsPt_All"),
      EfficiencyMEname = cms.untracked.string("HLT_Mu5/probeAnyMuon/effVsPt"),
    ),
    JpsiPars.clone(
      NumeratorMEname = cms.untracked.string("HLT_Mu5/probeAnyMuon/diMuonMassVsEta_L3Filtered"),
      DenominatorMEname = cms.untracked.string("HLT_Mu5/probeAnyMuon/diMuonMassVsEta_All"),
      EfficiencyMEname = cms.untracked.string("HLT_Mu5/probeAnyMuon/effVsEta"),
    ),
    JpsiPars.clone(
      NumeratorMEname = cms.untracked.string("HLT_Mu5/probeAnyMuon/diMuonMassVsPhi_L3Filtered"),
      DenominatorMEname = cms.untracked.string("HLT_Mu5/probeAnyMuon/diMuonMassVsPhi_All"),
      EfficiencyMEname = cms.untracked.string("HLT_Mu5/probeAnyMuon/effVsPhi"),
    ),
    JpsiPars.clone(
      NumeratorMEname = cms.untracked.string("HLT_Mu9/probeAnyMuon/diMuonMassVsPt_L3Filtered"),
      DenominatorMEname = cms.untracked.string("HLT_Mu9/probeAnyMuon/diMuonMassVsPt_All"),
      EfficiencyMEname = cms.untracked.string("HLT_Mu9/probeAnyMuon/effVsPt"),
    ),
    JpsiPars.clone(
      NumeratorMEname = cms.untracked.string("HLT_Mu9/probeAnyMuon/diMuonMassVsEta_L3Filtered"),
      DenominatorMEname = cms.untracked.string("HLT_Mu9/probeAnyMuon/diMuonMassVsEta_All"),
      EfficiencyMEname = cms.untracked.string("HLT_Mu9/probeAnyMuon/effVsEta"),
    ),
    JpsiPars.clone(
      NumeratorMEname = cms.untracked.string("HLT_Mu9/probeAnyMuon/diMuonMassVsPhi_L3Filtered"),
      DenominatorMEname = cms.untracked.string("HLT_Mu9/probeAnyMuon/diMuonMassVsPhi_All"),
      EfficiencyMEname = cms.untracked.string("HLT_Mu9/probeAnyMuon/effVsPhi"),
    )
  )
)

JpsiPostProcessor = cms.Sequence(JpsiClient)

Zpars = cms.untracked.PSet(
    MassDimension = cms.untracked.int32(2),
    FitFunction = cms.untracked.string("GaussianPlusLinear"),
    ExpectedMean = cms.untracked.double(91.),
    ExpectedSigma = cms.untracked.double(1.),
    Width = cms.untracked.double(2.5),
    FitRangeLow = cms.untracked.double(65),
    FitRangeHigh = cms.untracked.double(115),
    SignalRangeLow = cms.untracked.double(81),
    SignalRangeHigh = cms.untracked.double(101)
)

Zclient = cms.EDAnalyzer("DQMGenericTnPClient",
  MyDQMrootFolder = cms.untracked.string("HLT/Muon/Distributions"),
  # Set this if you want to save info about each fit
  #SavePlotsInRootFileName = cms.untracked.string("fittingPlots.root"),
  Verbose = cms.untracked.bool(False),
  Efficiencies = cms.untracked.VPSet(
    Zpars.clone(
      NumeratorMEname = cms.untracked.string("HLT_Mu3/probeAnyMuon/diMuonMassVsPt_L3Filtered"),
      DenominatorMEname = cms.untracked.string("HLT_Mu3/probeAnyMuon/diMuonMassVsPt_All"),
      EfficiencyMEname = cms.untracked.string("HLT_Mu3/probeAnyMuon/effVsPt"),
    ),
    Zpars.clone(
      NumeratorMEname = cms.untracked.string("HLT_Mu3/probeAnyMuon/diMuonMassVsEta_L3Filtered"),
      DenominatorMEname = cms.untracked.string("HLT_Mu3/probeAnyMuon/diMuonMassVsEta_All"),
      EfficiencyMEname = cms.untracked.string("HLT_Mu3/probeAnyMuon/effVsEta"),
    ),
    Zpars.clone(
      NumeratorMEname = cms.untracked.string("HLT_Mu3/probeAnyMuon/diMuonMassVsPhi_L3Filtered"),
      DenominatorMEname = cms.untracked.string("HLT_Mu3/probeAnyMuon/diMuonMassVsPhi_All"),
      EfficiencyMEname = cms.untracked.string("HLT_Mu3/probeAnyMuon/effVsPhi"),
    ),
    Zpars.clone(
      NumeratorMEname = cms.untracked.string("HLT_Mu5/probeAnyMuon/diMuonMassVsPt_L3Filtered"),
      DenominatorMEname = cms.untracked.string("HLT_Mu5/probeAnyMuon/diMuonMassVsPt_All"),
      EfficiencyMEname = cms.untracked.string("HLT_Mu5/probeAnyMuon/effVsPt"),
    ),
    Zpars.clone(
      NumeratorMEname = cms.untracked.string("HLT_Mu5/probeAnyMuon/diMuonMassVsEta_L3Filtered"),
      DenominatorMEname = cms.untracked.string("HLT_Mu5/probeAnyMuon/diMuonMassVsEta_All"),
      EfficiencyMEname = cms.untracked.string("HLT_Mu5/probeAnyMuon/effVsEta"),
    ),
    Zpars.clone(
      NumeratorMEname = cms.untracked.string("HLT_Mu5/probeAnyMuon/diMuonMassVsPhi_L3Filtered"),
      DenominatorMEname = cms.untracked.string("HLT_Mu5/probeAnyMuon/diMuonMassVsPhi_All"),
      EfficiencyMEname = cms.untracked.string("HLT_Mu5/probeAnyMuon/effVsPhi"),
    ),
    Zpars.clone(
      NumeratorMEname = cms.untracked.string("HLT_Mu9/probeAnyMuon/diMuonMassVsPt_L3Filtered"),
      DenominatorMEname = cms.untracked.string("HLT_Mu9/probeAnyMuon/diMuonMassVsPt_All"),
      EfficiencyMEname = cms.untracked.string("HLT_Mu9/probeAnyMuon/effVsPt"),
    ),
    Zpars.clone(
      NumeratorMEname = cms.untracked.string("HLT_Mu9/probeAnyMuon/diMuonMassVsEta_L3Filtered"),
      DenominatorMEname = cms.untracked.string("HLT_Mu9/probeAnyMuon/diMuonMassVsEta_All"),
      EfficiencyMEname = cms.untracked.string("HLT_Mu9/probeAnyMuon/effVsEta"),
    ),
    Zpars.clone(
      NumeratorMEname = cms.untracked.string("HLT_Mu9/probeAnyMuon/diMuonMassVsPhi_L3Filtered"),
      DenominatorMEname = cms.untracked.string("HLT_Mu9/probeAnyMuon/diMuonMassVsPhi_All"),
      EfficiencyMEname = cms.untracked.string("HLT_Mu9/probeAnyMuon/effVsPhi"),
    )
  )
)

ZPostProcessor = cms.Sequence(Zclient)

