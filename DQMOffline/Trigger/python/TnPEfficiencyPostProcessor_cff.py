import FWCore.ParameterSet.Config as cms

JpsiPars = cms.untracked.PSet(
      MassDimension = cms.untracked.int32(2),
      FitFunction = cms.untracked.string("GaussianPlusLinear"),
      ExpectedMean = cms.untracked.double(3.1),
      ExpectedSigma = cms.untracked.double(0.03),
      Width = cms.untracked.double(0),
      FitRangeLow = cms.untracked.double(2.8),
      FitRangeHigh = cms.untracked.double(3.4),
      SignalRangeLow = cms.untracked.double(2.95),
      SignalRangeHigh = cms.untracked.double(3.25)
)

jpsiClient = cms.EDAnalyzer("DQMGenericTnPClient",
  MyDQMrootFolder = cms.untracked.string("HLT/Muon/Distributions"),
  # Set this if you want to save info about each fit
  #SavePlotsInRootFileName = cms.untracked.string("fittingPlots.root"),
  SavePlotsInRootFileName = cms.untracked.string(""),						 
  Verbose = cms.untracked.bool(False),
  Efficiencies = cms.untracked.VPSet(
    JpsiPars.clone(
      NumeratorMEname = cms.untracked.string("HLT_Mu3/probeJpsiMuon/diMuonMassVsPt_L3Filtered"),
      DenominatorMEname = cms.untracked.string("HLT_Mu3/probeJpsiMuon/diMuonMassVsPt_All"),
      EfficiencyMEname = cms.untracked.string("HLT_Mu3/probeJpsiMuon/effVsPt"),
    ),
    JpsiPars.clone(
      NumeratorMEname = cms.untracked.string("HLT_Mu3/probeJpsiMuon/diMuonMassVsEta_L3Filtered"),
      DenominatorMEname = cms.untracked.string("HLT_Mu3/probeJpsiMuon/diMuonMassVsEta_All"),
      EfficiencyMEname = cms.untracked.string("HLT_Mu3/probeJpsiMuon/effVsEta"),
    ),
    JpsiPars.clone(
      NumeratorMEname = cms.untracked.string("HLT_Mu3/probeJpsiMuon/diMuonMassVsPhi_L3Filtered"),
      DenominatorMEname = cms.untracked.string("HLT_Mu3/probeJpsiMuon/diMuonMassVsPhi_All"),
      EfficiencyMEname = cms.untracked.string("HLT_Mu3/probeJpsiMuon/effVsPhi"),
    ),
    JpsiPars.clone(
      NumeratorMEname = cms.untracked.string("HLT_Mu5/probeJpsiMuon/diMuonMassVsPt_L3Filtered"),
      DenominatorMEname = cms.untracked.string("HLT_Mu5/probeJpsiMuon/diMuonMassVsPt_All"),
      EfficiencyMEname = cms.untracked.string("HLT_Mu5/probeJpsiMuon/effVsPt"),
    ),
    JpsiPars.clone(
      NumeratorMEname = cms.untracked.string("HLT_Mu5/probeJpsiMuon/diMuonMassVsEta_L3Filtered"),
      DenominatorMEname = cms.untracked.string("HLT_Mu5/probeJpsiMuon/diMuonMassVsEta_All"),
      EfficiencyMEname = cms.untracked.string("HLT_Mu5/probeJpsiMuon/effVsEta"),
    ),
    JpsiPars.clone(
      NumeratorMEname = cms.untracked.string("HLT_Mu5/probeJpsiMuon/diMuonMassVsPhi_L3Filtered"),
      DenominatorMEname = cms.untracked.string("HLT_Mu5/probeJpsiMuon/diMuonMassVsPhi_All"),
      EfficiencyMEname = cms.untracked.string("HLT_Mu5/probeJpsiMuon/effVsPhi"),
    ),
    JpsiPars.clone(
      NumeratorMEname = cms.untracked.string("HLT_Mu9/probeJpsiMuon/diMuonMassVsPt_L3Filtered"),
      DenominatorMEname = cms.untracked.string("HLT_Mu9/probeJpsiMuon/diMuonMassVsPt_All"),
      EfficiencyMEname = cms.untracked.string("HLT_Mu9/probeJpsiMuon/effVsPt"),
    ),
    JpsiPars.clone(
      NumeratorMEname = cms.untracked.string("HLT_Mu9/probeJpsiMuon/diMuonMassVsEta_L3Filtered"),
      DenominatorMEname = cms.untracked.string("HLT_Mu9/probeJpsiMuon/diMuonMassVsEta_All"),
      EfficiencyMEname = cms.untracked.string("HLT_Mu9/probeJpsiMuon/effVsEta"),
    ),
    JpsiPars.clone(
      NumeratorMEname = cms.untracked.string("HLT_Mu9/probeJpsiMuon/diMuonMassVsPhi_L3Filtered"),
      DenominatorMEname = cms.untracked.string("HLT_Mu9/probeJpsiMuon/diMuonMassVsPhi_All"),
      EfficiencyMEname = cms.untracked.string("HLT_Mu9/probeJpsiMuon/effVsPhi"),
    )
  )
)

UpsilonPars = cms.untracked.PSet(
      MassDimension = cms.untracked.int32(2),
      FitFunction = cms.untracked.string("GaussianPlusLinear"),
      ExpectedMean = cms.untracked.double(9.45),
      ExpectedSigma = cms.untracked.double(0.05),
      Width = cms.untracked.double(0),
      FitRangeLow = cms.untracked.double(8.5),
      FitRangeHigh = cms.untracked.double(10.5),
      SignalRangeLow = cms.untracked.double(9.),
      SignalRangeHigh = cms.untracked.double(10.)
)

upsilonClient = cms.EDAnalyzer("DQMGenericTnPClient",
  MyDQMrootFolder = cms.untracked.string("HLT/Muon/Distributions"),
  # Set this if you want to save info about each fit
  #SavePlotsInRootFileName = cms.untracked.string("fittingPlots.root"),
  SavePlotsInRootFileName = cms.untracked.string(""),
  Verbose = cms.untracked.bool(False),
  Efficiencies = cms.untracked.VPSet(
    UpsilonPars.clone(
      NumeratorMEname = cms.untracked.string("HLT_Mu3/probeUpsilonMuon/diMuonMassVsPt_L3Filtered"),
      DenominatorMEname = cms.untracked.string("HLT_Mu3/probeUpsilonMuon/diMuonMassVsPt_All"),
      EfficiencyMEname = cms.untracked.string("HLT_Mu3/probeUpsilonMuon/effVsPt"),
    ),
    UpsilonPars.clone(
      NumeratorMEname = cms.untracked.string("HLT_Mu3/probeUpsilonMuon/diMuonMassVsEta_L3Filtered"),
      DenominatorMEname = cms.untracked.string("HLT_Mu3/probeUpsilonMuon/diMuonMassVsEta_All"),
      EfficiencyMEname = cms.untracked.string("HLT_Mu3/probeUpsilonMuon/effVsEta"),
    ),
    UpsilonPars.clone(
      NumeratorMEname = cms.untracked.string("HLT_Mu3/probeUpsilonMuon/diMuonMassVsPhi_L3Filtered"),
      DenominatorMEname = cms.untracked.string("HLT_Mu3/probeUpsilonMuon/diMuonMassVsPhi_All"),
      EfficiencyMEname = cms.untracked.string("HLT_Mu3/probeUpsilonMuon/effVsPhi"),
    ),
    UpsilonPars.clone(
      NumeratorMEname = cms.untracked.string("HLT_Mu5/probeUpsilonMuon/diMuonMassVsPt_L3Filtered"),
      DenominatorMEname = cms.untracked.string("HLT_Mu5/probeUpsilonMuon/diMuonMassVsPt_All"),
      EfficiencyMEname = cms.untracked.string("HLT_Mu5/probeUpsilonMuon/effVsPt"),
    ),
    UpsilonPars.clone(
      NumeratorMEname = cms.untracked.string("HLT_Mu5/probeUpsilonMuon/diMuonMassVsEta_L3Filtered"),
      DenominatorMEname = cms.untracked.string("HLT_Mu5/probeUpsilonMuon/diMuonMassVsEta_All"),
      EfficiencyMEname = cms.untracked.string("HLT_Mu5/probeUpsilonMuon/effVsEta"),
    ),
    UpsilonPars.clone(
      NumeratorMEname = cms.untracked.string("HLT_Mu5/probeUpsilonMuon/diMuonMassVsPhi_L3Filtered"),
      DenominatorMEname = cms.untracked.string("HLT_Mu5/probeUpsilonMuon/diMuonMassVsPhi_All"),
      EfficiencyMEname = cms.untracked.string("HLT_Mu5/probeUpsilonMuon/effVsPhi"),
    ),
    UpsilonPars.clone(
      NumeratorMEname = cms.untracked.string("HLT_Mu9/probeUpsilonMuon/diMuonMassVsPt_L3Filtered"),
      DenominatorMEname = cms.untracked.string("HLT_Mu9/probeUpsilonMuon/diMuonMassVsPt_All"),
      EfficiencyMEname = cms.untracked.string("HLT_Mu9/probeUpsilonMuon/effVsPt"),
    ),
    UpsilonPars.clone(
      NumeratorMEname = cms.untracked.string("HLT_Mu9/probeUpsilonMuon/diMuonMassVsEta_L3Filtered"),
      DenominatorMEname = cms.untracked.string("HLT_Mu9/probeUpsilonMuon/diMuonMassVsEta_All"),
      EfficiencyMEname = cms.untracked.string("HLT_Mu9/probeUpsilonMuon/effVsEta"),
    ),
    UpsilonPars.clone(
      NumeratorMEname = cms.untracked.string("HLT_Mu9/probeUpsilonMuon/diMuonMassVsPhi_L3Filtered"),
      DenominatorMEname = cms.untracked.string("HLT_Mu9/probeUpsilonMuon/diMuonMassVsPhi_All"),
      EfficiencyMEname = cms.untracked.string("HLT_Mu9/probeUpsilonMuon/effVsPhi"),
    )
  )
)

ZPars = cms.untracked.PSet(
    MassDimension = cms.untracked.int32(2),
    FitFunction = cms.untracked.string("VoigtianPlusExponential"),
    ExpectedMean = cms.untracked.double(91.),
    ExpectedSigma = cms.untracked.double(1.),
    FixedWidth = cms.untracked.double(2.5),
    FitRangeLow = cms.untracked.double(65),
    FitRangeHigh = cms.untracked.double(115),
    SignalRangeLow = cms.untracked.double(81),
    SignalRangeHigh = cms.untracked.double(101)
)

zClient = cms.EDAnalyzer("DQMGenericTnPClient",
  MyDQMrootFolder = cms.untracked.string("HLT/Muon/Distributions"),
  # Set this if you want to save info about each fit
  #SavePlotsInRootFileName = cms.untracked.string("fittingPlots.root"),
  Verbose = cms.untracked.bool(False),
  Efficiencies = cms.untracked.VPSet(
    ZPars.clone(
      NumeratorMEname = cms.untracked.string("HLT_Mu3/probeZMuon/diMuonMassVsPt_L3Filtered"),
      DenominatorMEname = cms.untracked.string("HLT_Mu3/probeZMuon/diMuonMassVsPt_All"),
      EfficiencyMEname = cms.untracked.string("HLT_Mu3/probeZMuon/effVsPt"),
    ),
    ZPars.clone(
      NumeratorMEname = cms.untracked.string("HLT_Mu3/probeZMuon/diMuonMassVsEta_L3Filtered"),
      DenominatorMEname = cms.untracked.string("HLT_Mu3/probeZMuon/diMuonMassVsEta_All"),
      EfficiencyMEname = cms.untracked.string("HLT_Mu3/probeZMuon/effVsEta"),
    ),
    ZPars.clone(
      NumeratorMEname = cms.untracked.string("HLT_Mu3/probeZMuon/diMuonMassVsPhi_L3Filtered"),
      DenominatorMEname = cms.untracked.string("HLT_Mu3/probeZMuon/diMuonMassVsPhi_All"),
      EfficiencyMEname = cms.untracked.string("HLT_Mu3/probeZMuon/effVsPhi"),
    ),
    ZPars.clone(
      NumeratorMEname = cms.untracked.string("HLT_Mu5/probeZMuon/diMuonMassVsPt_L3Filtered"),
      DenominatorMEname = cms.untracked.string("HLT_Mu5/probeZMuon/diMuonMassVsPt_All"),
      EfficiencyMEname = cms.untracked.string("HLT_Mu5/probeZMuon/effVsPt"),
    ),
    ZPars.clone(
      NumeratorMEname = cms.untracked.string("HLT_Mu5/probeZMuon/diMuonMassVsEta_L3Filtered"),
      DenominatorMEname = cms.untracked.string("HLT_Mu5/probeZMuon/diMuonMassVsEta_All"),
      EfficiencyMEname = cms.untracked.string("HLT_Mu5/probeZMuon/effVsEta"),
    ),
    ZPars.clone(
      NumeratorMEname = cms.untracked.string("HLT_Mu5/probeZMuon/diMuonMassVsPhi_L3Filtered"),
      DenominatorMEname = cms.untracked.string("HLT_Mu5/probeZMuon/diMuonMassVsPhi_All"),
      EfficiencyMEname = cms.untracked.string("HLT_Mu5/probeZMuon/effVsPhi"),
    ),
    ZPars.clone(
      NumeratorMEname = cms.untracked.string("HLT_Mu9/probeZMuon/diMuonMassVsPt_L3Filtered"),
      DenominatorMEname = cms.untracked.string("HLT_Mu9/probeZMuon/diMuonMassVsPt_All"),
      EfficiencyMEname = cms.untracked.string("HLT_Mu9/probeZMuon/effVsPt"),
    ),
    ZPars.clone(
      NumeratorMEname = cms.untracked.string("HLT_Mu9/probeZMuon/diMuonMassVsEta_L3Filtered"),
      DenominatorMEname = cms.untracked.string("HLT_Mu9/probeZMuon/diMuonMassVsEta_All"),
      EfficiencyMEname = cms.untracked.string("HLT_Mu9/probeZMuon/effVsEta"),
    ),
    ZPars.clone(
      NumeratorMEname = cms.untracked.string("HLT_Mu9/probeZMuon/diMuonMassVsPhi_L3Filtered"),
      DenominatorMEname = cms.untracked.string("HLT_Mu9/probeZMuon/diMuonMassVsPhi_All"),
      EfficiencyMEname = cms.untracked.string("HLT_Mu9/probeZMuon/effVsPhi"),
    )
  )
)

tagAndProbeEfficiencyPostProcessor = cms.Sequence(jpsiClient*upsilonClient*zClient)

