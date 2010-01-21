import FWCore.ParameterSet.Config as cms
import copy

process = cms.Process("ZToMuMuHistos")

process.include("FWCore/MessageLogger/data/MessageLogger.cfi")

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(
    "file:/scratch1/cms/data/summer08/skim/dimuons_skim_zmumu.root"
    )
)

zSelection = cms.PSet(
    cut = cms.string("charge = 0 & daughter(0).pt > 20 & daughter(1).pt > 20 & abs(daughter(0).eta)<2 & abs(daughter(1).eta)<2 & mass > 20"),
    isoCut = cms.double(100.0),
    isolationType = cms.string("track"),
)

process.goodZToMuMu = cms.EDFilter(
    "ZToMuMuIsolatedSelector",
    zSelection,
    src = cms.InputTag("dimuonsGlobal"),
    filter = cms.bool(True) 
)

process.nonIsolatedZToMuMu = cms.EDFilter(
    "ZToMuMuNonIsolatedSelector",
    zSelection,
    src = cms.InputTag("dimuonsGlobal"),
    filter = cms.bool(True) 
)

process.zToMuGlobalMuOneTrack = cms.EDFilter(
    "CandViewRefSelector",
    cut = cms.string("daughter(0).isGlobalMuon = 1"),
    src = cms.InputTag("dimuonsOneTrack"),
    filter = cms.bool(True)
)

process.zToMuMuOneTrack = cms.EDFilter(
    "ZToMuMuIsolatedSelector",
    zSelection,
    src = cms.InputTag("zToMuGlobalMuOneTrack"),
    filter = cms.bool(True)
)

process.zToMuMuOneStandAloneMuon = cms.EDFilter(
    "ZToMuMuIsolatedSelector",
    zSelection,
    src = cms.InputTag("dimuonsOneStandAloneMuon"),
    filter = cms.bool(True)
)

process.goodZToMuMuOneTrack = cms.EDFilter(
    "ZMuMuOverlapExclusionSelector",
    src = cms.InputTag("zToMuMuOneTrack"),
    overlap = cms.InputTag("goodZToMuMu"),
    filter = cms.bool(True)
)

process.goodZToMuMuOneStandAloneMuon = cms.EDFilter(
    "ZMuMuOverlapExclusionSelector",
    src = cms.InputTag("zToMuMuOneStandAloneMuon"),
    overlap = cms.InputTag("goodZToMuMu"),
    filter = cms.bool(True)
)

process.eventInfo = cms.OutputModule (
    "AsciiOutputModule"
)

process.out = cms.OutputModule(
    "PoolOutputModule",
    fileName = cms.untracked.string('file:./Zmm.root'),
    outputCommands = cms.untracked.vstring(
      "drop *",
#      "keep *_genParticles_*_*",
#      "keep L1MuRegionalCands_*_*_*",
#      "keep L1MuGMTCands_*_*_*",
#      "keep L1MuGMTReadoutCollection_*_*_*",
      "keep TriggerResults_*_*_*",
#      "keep recoTracks_generalTracks_*_*",
#      "keep recoTracks_globalMuons_*_*",
#      "keep recoTracks_standAloneMuons_*_*",
#      "keep recoMuons_muons_*_*",
      "keep *_selectedLayer1Muons_*_*",
      "keep *_selectedLayer1TrackCands_*_*",
      'keep *_selectedLayer1MuonsTriggerMatch_*_*',
#     "keep *_goodTracks_*_*",
      "keep *_dimuons_*_*",
      "keep *_dimuonsOneTrack_*_*",
      "keep *_dimuonsGlobal_*_*",
      "keep *_dimuonsOneStandAloneMuon_*_*",
#      "keep *_muonMatch_*_*",
#      "keep *_allDimuonsMCMatch_*_*",
      "keep *_goodZToMuMu__*",
      "keep *_nonIsolatedZToMuMu__*",
      "keep *_goodZToMuMuOneStandAloneMuon__*",
      "keep *_goodZToMuMuOneTrack__*"
    ),
    SelectEvents = cms.untracked.PSet(
      SelectEvents = cms.vstring(
        "goodZToMuMuPath", "nonIsolatedZToMuMuPath",
        "zToMuMuOneStandAloneMuonPath", "goodZToMuMuOneTrackPath"
      )
    )
)

process.goodZToMuMuPath = cms.Path(
    process.goodZToMuMu 
)
    
process.nonIsolatedZToMuMuPath = cms.Path(
    process.nonIsolatedZToMuMu
)

process.zToMuMuOneStandAloneMuonPath = cms.Path(
    ~process.goodZToMuMu + 
    process.zToMuMuOneStandAloneMuon + 
    process.goodZToMuMuOneStandAloneMuon 
)

process.goodZToMuMuOneTrackPath = cms.Path(
    ~process.goodZToMuMu +
    ~process.zToMuMuOneStandAloneMuon +
    process.zToMuGlobalMuOneTrack +
    process.zToMuMuOneTrack +
    process.goodZToMuMuOneTrack 
)


process.endPath = cms.EndPath( 
    process.eventInfo +
    process.out
)

