import FWCore.ParameterSet.Config as cms

process = cms.Process("ZMuMuEfficiencyAnalyzer")
process.load("ElectroWeakAnalysis.ZReco.mcTruthForDimuons_cff")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:/scratch1/cms/data/summer08/skim/dimuons_skim_zmumu.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.TFileService = cms.Service("TFileService",
    fileName = cms.string('zMuMu_efficiencyAnalyzer.root')
)

process.zToMuMu = cms.EDFilter("CandViewRefSelector",
    src = cms.InputTag("dimuons"),
    cut = cms.string('daughter(0).isGlobalMuon = 1 & daughter(1).isGlobalMuon = 1'),
)

process.goodZToMuMuOneStandAloneMuon = cms.EDFilter(
    "ZMuMuOverlapExclusionSelector",
    src = cms.InputTag("dimuonsOneStandAloneMuon"),
    overlap = cms.InputTag("zToMuMu"),
)

process.zToMuMuOneTrack = cms.EDFilter(
    "CandViewRefSelector",
    src = cms.InputTag("dimuonsOneTrack"),
    cut = cms.string('daughter(0).isGlobalMuon = 1'),
)

process.goodZToMuMuOneTrack = cms.EDFilter(
    "ZMuMuOverlapExclusionSelector",
    src = cms.InputTag("zToMuMuOneTrack"),
    overlap = cms.InputTag("zToMuMu"),
)


process.zMuMu_efficiencyAnalyzer = cms.EDFilter("ZMuMu_efficiencyAnalyzer",
    muons = cms.InputTag("selectedLayer1Muons"),
    tracks = cms.InputTag("selectedLayer1TrackCands"),
    zMuMu = cms.InputTag("zToMuMu"),
    zMuStandAlone = cms.InputTag("goodZToMuMuOneStandAloneMuon"),
    zMuTrack = cms.InputTag("goodZToMuMuOneTrack"),
    zMuMuMatchMap = cms.InputTag("allDimuonsMCMatch"),
    zMuStandAloneMatchMap = cms.InputTag("allDimuonsMCMatch"),
    zMuTrackMatchMap = cms.InputTag("allDimuonsMCMatch"),
    genParticles = cms.InputTag("genParticles"),
    primaryVertices =cms.InputTag("offlinePrimaryVertices"),
    bothMuons = cms.bool(True),                              
    zMassMin = cms.untracked.double(20.0),
    zMassMax = cms.untracked.double(200.0),
    isomax = cms.untracked.double(3.0),
    etamax = cms.untracked.double(2.0),
    ptmin = cms.untracked.double(20.0),
)

process.eventInfo = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.mcTruthForDimuons *
                     process.zToMuMu *
                     process.goodZToMuMuOneStandAloneMuon *                     
                     process.zToMuMuOneTrack *
                     process.goodZToMuMuOneTrack *                     
                     process.zMuMu_efficiencyAnalyzer)
process.e = cms.EndPath(process.eventInfo)

