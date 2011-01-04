import FWCore.ParameterSet.Config as cms

hltMuonEfficiencies = cms.EDAnalyzer("DQMGenericClient",

    subDirs        = cms.untracked.vstring("HLT/Muon/Distributions/*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    outputFileName = cms.untracked.string(''),
    commands       = cms.vstring(),
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(),

    efficiencyProfile = cms.untracked.vstring(
        "efficiencyEta 'Efficiency to Match Reco Muons to Trigger Objects; #eta^{reco}; N(#mu matched to trigger object) / N(#mu)' efficiencyEta_numer efficiencyEta_denom",
        "efficiencyPhi 'Efficiency to Match Reco Muons to Trigger Objects; #phi^{reco}; N(#mu matched to trigger object) / N(#mu)' efficiencyPhi_numer efficiencyPhi_denom",
        "efficiencyTurnOn 'Efficiency to Match Reco Muons to Trigger Objects; p_{T}^{reco}; N(#mu matched to trigger object) / N(#mu)' efficiencyTurnOn_numer efficiencyTurnOn_denom",
        "efficiencyD0 'Efficiency to Match Reco Muons to Trigger Objects; d0^{reco}; N(#mu matched to trigger object) / N(#mu)' efficiencyD0_numer efficiencyD0_denom",
        "efficiencyZ0 'Efficiency to Match Reco Muons to Trigger Objects; z0^{reco}; N(#mu matched to trigger object) / N(#mu)' efficiencyZ0_numer efficiencyZ0_denom",
        "efficiencyCharge 'Efficiency to Match Reco Muons to Trigger Objects; q^{reco}; N(#mu matched to trigger object) / N(#mu)' efficiencyCharge_numer efficiencyCharge_denom",
        "fakerateEta 'Trigger Fake Rate; #eta^{trigger}; N(unmatched trigger objects) / N(trigger objects)' fakerateEta_numer fakerateEta_denom",
        "fakeratePhi 'Trigger Fake Rate; #phi^{trigger}; N(unmatched trigger objects) / N(trigger objects)' fakeratePhi_numer fakeratePhi_denom",
        "fakerateTurnOn 'Trigger Fake Rate; p_{T}^{trigger}; N(unmatched trigger objects) / N(trigger objects)' fakerateTurnOn_numer fakerateTurnOn_denom",
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
    SignalRangeHigh = cms.untracked.double(101),
)

zClient = cms.EDAnalyzer("DQMGenericTnPClient",
  subDirs = cms.untracked.vstring("HLT/Muon/Distributions/*"),
  #MyDQMrootFolder = cms.untracked.string("HLT/Muon/Distributions/vbtfMuons/HLT_Mu5"),
  # Set this if you want to save info about each fit
  # SavePlotsInRootFileName = cms.untracked.string("fittingPlots.root"),
  Verbose = cms.untracked.bool(False),
  Efficiencies = cms.untracked.VPSet(
    ZPars.clone(
      NumeratorMEname = cms.untracked.string("massVsEta_numer"),
      DenominatorMEname = cms.untracked.string("massVsEta_denom"),
      EfficiencyMEname = cms.untracked.string("massVsEta_efficiency"),
    ),
  )
)


hltMuonPostVal = cms.Sequence(
    hltMuonEfficiencies *
    zClient
)
