import FWCore.ParameterSet.Config as cms

DQMGenericTnPClientPars = cms.untracked.PSet(
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

DQMGenericTnPClient = cms.EDAnalyzer("DQMGenericTnPClient",
  MyDQMrootFolder = cms.untracked.string("HLT/Muon/"),
  # Set this if you want to save the fitting plots
  #SavePlotsInRootFileName = cms.untracked.string("fittingPlots.root"),
  Verbose = cms.untracked.bool(False),
  Efficiencies = cms.untracked.VPSet(
    DQMGenericTnPClientPars.clone(
      NumeratorMEname = cms.untracked.string("globalMuons"),
      DenominatorMEname = cms.untracked.string("tracks"),
      EfficiencyMEname = cms.untracked.string("effGlobal")
    ),
  )
)

