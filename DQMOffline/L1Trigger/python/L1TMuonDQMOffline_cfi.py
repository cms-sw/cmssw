import FWCore.ParameterSet.Config as cms
import math

muonEfficiencyThresholds = [16, 20, 25]

# define binning for efficiency plots
# pt
effVsPtBins = range(0, 50, 2)
effVsPtBins += range(50, 70, 5)
effVsPtBins += range(70, 100, 10)
effVsPtBins += range(100, 200, 25)
effVsPtBins += range(200, 300, 50)
effVsPtBins += range(300, 500, 100)
effVsPtBins.append(500)

# phi
nPhiBins = 24
phiMin = -math.pi
phiMax = math.pi
effVsPhiBins = [i*(phiMax-phiMin)/nPhiBins + phiMin for i in range(nPhiBins+1)]

# eta
nEtaBins = 50
etaMin = -2.5
etaMax = 2.5
effVsEtaBins = [i*(etaMax-etaMin)/nEtaBins + etaMin for i in range(nEtaBins+1)]

l1tMuonDQMOffline = cms.EDAnalyzer("L1TMuonDQMOffline",
    histFolder = cms.untracked.string('L1T/L1TMuon'),
    gmtPtCuts = cms.untracked.vint32(muonEfficiencyThresholds),
    tagPtCut = cms.untracked.double(30.),
    muonInputTag = cms.untracked.InputTag("muons"),
    gmtInputTag  = cms.untracked.InputTag("gmtStage2Digis","Muon"),
    vtxInputTag = cms.untracked.InputTag("offlinePrimaryVertices"),
    bsInputTag  = cms.untracked.InputTag("offlineBeamSpot"),

    triggerNames = cms.untracked.vstring(
        "HLT_IsoMu18_v*",
        "HLT_IsoMu20_v*",
        "HLT_IsoMu22_v*",
        "HLT_IsoMu24_v*",
        "HLT_IsoMu27_v*",
        "HLT_Mu30_v*",
        "HLT_Mu40_v*"
    ),
    trigInputTag       = cms.untracked.InputTag("hltTriggerSummaryAOD", "", "HLT"),
    trigProcess        = cms.untracked.string("HLT"),
    trigProcess_token  = cms.untracked.InputTag("TriggerResults","","HLT"),

    efficiencyVsPtBins = cms.untracked.vdouble(effVsPtBins),
    efficiencyVsPhiBins = cms.untracked.vdouble(effVsPhiBins),
    efficiencyVsEtaBins = cms.untracked.vdouble(effVsEtaBins),

    verbose   = cms.untracked.bool(False)
)
