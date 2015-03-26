import FWCore.ParameterSet.Config as cms

heavyFlavorValidation = cms.EDAnalyzer("HeavyFlavorValidation",
    DQMFolder = cms.untracked.string("HLT/HeavyFlavor"),
    TriggerProcessName = cms.untracked.string("HLT"),
    TriggerPathName = cms.untracked.string("HLT_Mu5"),
    TriggerSummaryRAW = cms.untracked.string("hltTriggerSummaryRAW"),
    TriggerSummaryAOD = cms.untracked.string("hltTriggerSummaryAOD"),
    TriggerResults = cms.untracked.string("TriggerResults"),
    RecoMuons = cms.InputTag("muons"),
    GenParticles = cms.InputTag("genParticles"),
# list IDs of muon mothers, -1:don't check, 0:particle gun, 23:Z, 443:J/psi, 553:Upsilon, 531:Bs, 333:Phi
    MotherIDs = cms.untracked.vint32(23,443,553,531,333,0),
    GenGlobDeltaRMatchingCut = cms.untracked.double(0.1),
    GlobL1DeltaRMatchingCut = cms.untracked.double(0.3),
    GlobL2DeltaRMatchingCut = cms.untracked.double(0.3),
    GlobL3DeltaRMatchingCut = cms.untracked.double(0.1),
    DeltaEtaBins = cms.untracked.vdouble(100, -.5, .5),
    DeltaPhiBins = cms.untracked.vdouble(100, -.5, .5),
    MuonPtBins = cms.untracked.vdouble(1., 3., 5., 9., 15., 32., 64., 128., 256., 512., 1024., 2048.),
    MuonEtaBins = cms.untracked.vdouble(16, -2.4, 2.4),
    MuonPhiBins = cms.untracked.vdouble(12, -3.15, 3.15),
    DimuonPtBins = cms.untracked.vdouble(0., 2., 4., 6., 8., 10., 15., 25., 50., 100.),
    DimuonEtaBins = cms.untracked.vdouble(16, -2.4, 2.4),
    DimuonDRBins = cms.untracked.vdouble(10, 0., 1.)
)
