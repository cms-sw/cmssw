import FWCore.ParameterSet.Config as cms

JpsiTrigEff = cms.EDAnalyzer("DQMGenericTnPClient",
  MyDQMrootFolder = cms.untracked.string("HLT/Muon/Distributions"),
  # Set this if you want to save info about each fit
  #SavePlotsInRootFileName = cms.untracked.string("fittingPlots.root"),
  SavePlotsInRootFileName = cms.untracked.string(""),						 
  Verbose = cms.untracked.bool(False),
  Efficiencies = cms.untracked.VPSet(
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("HLT_Mu3/probeAnyMuon/diMuonMassVsPt_L3Filtered"),
      DenominatorMEname = cms.untracked.string("HLT_Mu3/probeAnyMuon/diMuonMassVsPt_All"),
      EfficiencyMEname = cms.untracked.string("HLT_Mu3/probeAnyMuon/effVsPt"),
      MassDimension = cms.untracked.int32(2),
      FitFunction = cms.untracked.string("GaussianPlusLinear"),
      ExpectedMean = cms.untracked.double(3.1),
      ExpectedSigma = cms.untracked.double(0.005),
      Width = cms.untracked.double(0),
      FitRangeLow = cms.untracked.double(2.6),
      FitRangeHigh = cms.untracked.double(3.6),
      SignalRangeLow = cms.untracked.double(2.9),
      SignalRangeHigh = cms.untracked.double(3.3)
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("HLT_Mu3/probeAnyMuon/diMuonMassVsEta_L3Filtered"),
      DenominatorMEname = cms.untracked.string("HLT_Mu3/probeAnyMuon/diMuonMassVsEta_All"),
      EfficiencyMEname = cms.untracked.string("HLT_Mu3/probeAnyMuon/effVsEta"),
      MassDimension = cms.untracked.int32(2),
      FitFunction = cms.untracked.string("GaussianPlusLinear"),
      ExpectedMean = cms.untracked.double(3.1),
      ExpectedSigma = cms.untracked.double(0.005),
      Width = cms.untracked.double(0),
      FitRangeLow = cms.untracked.double(2.6),
      FitRangeHigh = cms.untracked.double(3.6),
      SignalRangeLow = cms.untracked.double(2.9),
      SignalRangeHigh = cms.untracked.double(3.3)
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("HLT_Mu3/probeAnyMuon/diMuonMassVsPhi_L3Filtered"),
      DenominatorMEname = cms.untracked.string("HLT_Mu3/probeAnyMuon/diMuonMassVsPhi_All"),
      EfficiencyMEname = cms.untracked.string("HLT_Mu3/probeAnyMuon/effVsPhi"),
      MassDimension = cms.untracked.int32(2),
      FitFunction = cms.untracked.string("GaussianPlusLinear"),
      ExpectedMean = cms.untracked.double(3.1),
      ExpectedSigma = cms.untracked.double(0.005),
      Width = cms.untracked.double(0),
      FitRangeLow = cms.untracked.double(2.6),
      FitRangeHigh = cms.untracked.double(3.6),
      SignalRangeLow = cms.untracked.double(2.9),
      SignalRangeHigh = cms.untracked.double(3.3)
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("HLT_Mu5/probeAnyMuon/diMuonMassVsPt_L3Filtered"),
      DenominatorMEname = cms.untracked.string("HLT_Mu5/probeAnyMuon/diMuonMassVsPt_All"),
      EfficiencyMEname = cms.untracked.string("HLT_Mu5/probeAnyMuon/effVsPt"),
      MassDimension = cms.untracked.int32(2),
      FitFunction = cms.untracked.string("GaussianPlusLinear"),
      ExpectedMean = cms.untracked.double(3.1),
      ExpectedSigma = cms.untracked.double(0.005),
      Width = cms.untracked.double(0),
      FitRangeLow = cms.untracked.double(2.6),
      FitRangeHigh = cms.untracked.double(3.6),
      SignalRangeLow = cms.untracked.double(2.9),
      SignalRangeHigh = cms.untracked.double(3.3)
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("HLT_Mu5/probeAnyMuon/diMuonMassVsEta_L3Filtered"),
      DenominatorMEname = cms.untracked.string("HLT_Mu5/probeAnyMuon/diMuonMassVsEta_All"),
      EfficiencyMEname = cms.untracked.string("HLT_Mu5/probeAnyMuon/effVsEta"),
      MassDimension = cms.untracked.int32(2),
      FitFunction = cms.untracked.string("GaussianPlusLinear"),
      ExpectedMean = cms.untracked.double(3.1),
      ExpectedSigma = cms.untracked.double(0.005),
      Width = cms.untracked.double(0),
      FitRangeLow = cms.untracked.double(2.6),
      FitRangeHigh = cms.untracked.double(3.6),
      SignalRangeLow = cms.untracked.double(2.9),
      SignalRangeHigh = cms.untracked.double(3.3)
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("HLT_Mu5/probeAnyMuon/diMuonMassVsPhi_L3Filtered"),
      DenominatorMEname = cms.untracked.string("HLT_Mu5/probeAnyMuon/diMuonMassVsPhi_All"),
      EfficiencyMEname = cms.untracked.string("HLT_Mu5/probeAnyMuon/effVsPhi"),
      MassDimension = cms.untracked.int32(2),
      FitFunction = cms.untracked.string("GaussianPlusLinear"),
      ExpectedMean = cms.untracked.double(3.1),
      ExpectedSigma = cms.untracked.double(0.005),
      Width = cms.untracked.double(0),
      FitRangeLow = cms.untracked.double(2.6),
      FitRangeHigh = cms.untracked.double(3.6),
      SignalRangeLow = cms.untracked.double(2.9),
      SignalRangeHigh = cms.untracked.double(3.3)
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("HLT_Mu9/probeAnyMuon/diMuonMassVsPt_L3Filtered"),
      DenominatorMEname = cms.untracked.string("HLT_Mu9/probeAnyMuon/diMuonMassVsPt_All"),
      EfficiencyMEname = cms.untracked.string("HLT_Mu9/probeAnyMuon/effVsPt"),
      MassDimension = cms.untracked.int32(2),
      FitFunction = cms.untracked.string("GaussianPlusLinear"),
      ExpectedMean = cms.untracked.double(3.1),
      ExpectedSigma = cms.untracked.double(0.005),
      Width = cms.untracked.double(0),
      FitRangeLow = cms.untracked.double(2.6),
      FitRangeHigh = cms.untracked.double(3.6),
      SignalRangeLow = cms.untracked.double(2.9),
      SignalRangeHigh = cms.untracked.double(3.3)
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("HLT_Mu9/probeAnyMuon/diMuonMassVsEta_L3Filtered"),
      DenominatorMEname = cms.untracked.string("HLT_Mu9/probeAnyMuon/diMuonMassVsEta_All"),
      EfficiencyMEname = cms.untracked.string("HLT_Mu9/probeAnyMuon/effVsEta"),
      MassDimension = cms.untracked.int32(2),
      FitFunction = cms.untracked.string("GaussianPlusLinear"),
      ExpectedMean = cms.untracked.double(3.1),
      ExpectedSigma = cms.untracked.double(0.005),
      Width = cms.untracked.double(0),
      FitRangeLow = cms.untracked.double(2.6),
      FitRangeHigh = cms.untracked.double(3.6),
      SignalRangeLow = cms.untracked.double(2.9),
      SignalRangeHigh = cms.untracked.double(3.3)
    ),
    cms.untracked.PSet(
      NumeratorMEname = cms.untracked.string("HLT_Mu9/probeAnyMuon/diMuonMassVsPhi_L3Filtered"),
      DenominatorMEname = cms.untracked.string("HLT_Mu9/probeAnyMuon/diMuonMassVsPhi_All"),
      EfficiencyMEname = cms.untracked.string("HLT_Mu9/probeAnyMuon/effVsPhi"),
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
  )
)

bPAGPostProcessor = cms.Sequence(JpsiTrigEff)
