import FWCore.ParameterSet.Config as cms

# define binning for efficiency plots
# pt
effVsPtBins = range(0, 30, 1)
effVsPtBins += range(30, 50, 2)
effVsPtBins += range(50, 70, 5)
effVsPtBins += range(70, 100, 10)
effVsPtBins += range(100, 200, 25)
effVsPtBins += range(200, 300, 50)
effVsPtBins += range(300, 500, 100)
effVsPtBins += range(500, 700, 200)
effVsPtBins += range(700, 1000, 300)
effVsPtBins.append(1000)

# phi
nPhiBins = 34
phiMin = -3.4
phiMax = 3.4
effVsPhiBins = [i*(phiMax-phiMin)/nPhiBins + phiMin for i in range(nPhiBins+1)]

# eta
nEtaBins = 50
etaMin = -2.5
etaMax = 2.5
effVsEtaBins = [i*(etaMax-etaMin)/nEtaBins + etaMin for i in range(nEtaBins+1)]

# vtx
effVsVtxBins = range(0, 101)

# A list of pt cut + quality cut pairs for which efficiency plots should be made
ptQualCuts = [[22, 12], [15, 8], [7, 8], [3, 4]]
cutsPSets = []
for ptQualCut in ptQualCuts:
    cutsPSets.append(cms.untracked.PSet(ptCut = cms.untracked.int32(ptQualCut[0]),
                                        qualCut = cms.untracked.int32(ptQualCut[1])))

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
l1tMuonDQMOffline = DQMEDAnalyzer('L1TMuonDQMOffline',
    histFolder = cms.untracked.string('L1T/L1TObjects/L1TMuon/L1TriggerVsReco'),
    tagPtCut = cms.untracked.double(26.),
    recoToL1PtCutFactor = cms.untracked.double(1.2),
    cuts = cms.untracked.VPSet(cutsPSets),
    useL1AtVtxCoord = cms.untracked.bool(True),

    muonInputTag = cms.untracked.InputTag("muons"),
    gmtInputTag  = cms.untracked.InputTag("gmtStage2Digis","Muon"),
    vtxInputTag = cms.untracked.InputTag("offlinePrimaryVertices"),
    bsInputTag  = cms.untracked.InputTag("offlineBeamSpot"),

    triggerNames = cms.untracked.vstring(
        "HLT_IsoMu27_v*",
        "HLT_IsoMu30_v*"
    ),
    trigInputTag       = cms.untracked.InputTag("hltTriggerSummaryAOD", "", "HLT"),
    trigProcess        = cms.untracked.string("HLT"),
    trigProcess_token  = cms.untracked.InputTag("TriggerResults","","HLT"),

    efficiencyVsPtBins = cms.untracked.vdouble(effVsPtBins),
    efficiencyVsPhiBins = cms.untracked.vdouble(effVsPhiBins),
    efficiencyVsEtaBins = cms.untracked.vdouble(effVsEtaBins),
    efficiencyVsVtxBins = cms.untracked.vdouble(effVsVtxBins),

    # muon track extrapolation to 2nd station
    muProp = cms.PSet(
        useTrack = cms.string("tracker"),  # 'none' to use Candidate P4; or 'tracker', 'muon', 'global'
        useState = cms.string("atVertex"), # 'innermost' and 'outermost' require the TrackExtra
        useSimpleGeometry = cms.bool(True),
        useStation2 = cms.bool(True),
        fallbackToME1 = cms.bool(False),
    ),

    verbose   = cms.untracked.bool(False)
)

# emulator module
l1tMuonDQMOfflineEmu = l1tMuonDQMOffline.clone(
    gmtInputTag  = cms.untracked.InputTag("simGmtStage2Digis"),
    histFolder = cms.untracked.string('L1TEMU/L1TObjects/L1TMuon/L1TriggerVsReco')
)

# modifications for the pp reference run
# A list of pt cut + quality cut pairs for which efficiency plots should be made
ptQualCuts_HI = [[12, 12], [7, 8], [5, 4]]
cutsPSets_HI = []
for ptQualCut in ptQualCuts_HI:
    cutsPSets_HI.append(cms.untracked.PSet(ptCut = cms.untracked.int32(ptQualCut[0]),
                                           qualCut = cms.untracked.int32(ptQualCut[1])))
from Configuration.Eras.Modifier_ppRef_2017_cff import ppRef_2017
ppRef_2017.toModify(l1tMuonDQMOffline,
    tagPtCut = cms.untracked.double(14.),
    cuts = cms.untracked.VPSet(cutsPSets_HI),
    triggerNames = cms.untracked.vstring(
        "HLT_HIL3Mu12_v*",
    )
)
ppRef_2017.toModify(l1tMuonDQMOfflineEmu,
    tagPtCut = cms.untracked.double(14.),
    cuts = cms.untracked.VPSet(cutsPSets_HI),
    triggerNames = cms.untracked.vstring(
        "HLT_HIL3Mu12_v*",
    )
)


