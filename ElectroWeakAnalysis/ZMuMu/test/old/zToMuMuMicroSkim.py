import FWCore.ParameterSet.Config as cms
import copy

process = cms.Process("reskim")

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(
    "file:/scratch1/users/fabozzi/patv2_skim/testSkim_v2.root"
    )
)

process.include("FWCore/MessageLogger/data/MessageLogger.cfi")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'IDEAL_V9::All'
process.load("Configuration.StandardSequences.MagneticField_cff")

process.prunedGenParticles = cms.EDProducer(
    "GenParticlePruner",
    src = cms.InputTag("genParticles"),
    select = cms.vstring(
    "drop  *  ", # this is the default
    "keep+ pdgId = {Z0}",
    "keep+ pdgId = {W-}",
    "keep++ pdgId = {mu-}"
    )
)

process.load("ElectroWeakAnalysis.ZReco.dimuonsHLTFilter_cfi")
process.load("ElectroWeakAnalysis.ZReco.patCandidatesForDimuonsSequences_cff")
process.load("ElectroWeakAnalysis.ZReco.dimuons_cfi")
process.load("ElectroWeakAnalysis.ZReco.dimuonsOneTrack_cfi")
process.load("ElectroWeakAnalysis.ZReco.dimuonsGlobal_cfi")
process.load("ElectroWeakAnalysis.ZReco.dimuonsOneStandAloneMuon_cfi")
process.load("ElectroWeakAnalysis.ZReco.mcTruthForDimuons_cff")
process.load("ElectroWeakAnalysis.ZReco.dimuonsFilter_cfi")
process.load("ElectroWeakAnalysis.ZReco.dimuonsOneTrackFilter_cfi")

process.muonMatch.matched = cms.InputTag("prunedGenParticles")
process.trackMuMatch.matched = cms.InputTag("prunedGenParticles")

zSelection = cms.PSet(
    cut = cms.string("charge = 0 & daughter(0).pt > 20 & daughter(1).pt > 20 & "
                     "abs(daughter(0).eta)<2 & abs(daughter(1).eta)<2 & mass > 20"),
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
    fileName = cms.untracked.string('file:zmm_reskim_v2.root'),
    outputCommands = cms.untracked.vstring(
      "drop *",
      "keep *_prunedGenParticles_*_reskim",
      "keep *_selectedLayer1Muons_*_reskim",
      "keep *_selectedLayer1TrackCands_*_reskim",
      "keep *_dimuons_*_reskim",
      "keep *_dimuonsOneTrack_*_reskim",
      "keep *_dimuonsGlobal_*_reskim",
      "keep *_dimuonsOneStandAloneMuon_*_reskim",
      "keep *_muonMatch_*_reskim",
      "keep *_allDimuonsMCMatch_*_reskim",
      "keep *_goodZToMuMu_*_reskim",
      "keep *_nonIsolatedZToMuMu_*_reskim",
      "keep *_goodZToMuMuOneStandAloneMuon_*_reskim",
      "keep *_goodZToMuMuOneTrack_*_reskim",
      # "keep *_genParticles_*_*",
      "keep L1MuRegionalCands_*_*_*",
      "keep L1MuGMTCands_*_*_*",
      "keep L1MuGMTReadoutCollection_*_*_*",
     # "keep l1extraL1MuonParticles_*_*_*",
      "keep TriggerResults_*_*_*",
      "keep recoTracks_generalTracks_*_*",
      "keep recoTracks_globalMuons_*_*",
      "keep recoTracks_standAloneMuons_*_*",
      "keep recoMuons_muons_*_*",
      "keep *_selectedLayer1Muons_*_*",
      "keep *_selectedLayer1TrackCands_*_*",
#      "keep *_goodTracks_*_*",
       "drop *_*_*_TestDimuonReco" 

    ),
    SelectEvents = cms.untracked.PSet(
      SelectEvents = cms.vstring(
        "goodZToMuMuPath", "nonIsolatedZToMuMuPath",
        "zToMuMuOneStandAloneMuonPath", "goodZToMuMuOneTrackPath"
      )
    )
)

process.genParticlesPath = cms.Path(
    process.prunedGenParticles
    )

process.dimuonsPath = cms.Path(
    process.dimuonsHLTFilter +
    process.goodMuonRecoForDimuon +
    process.dimuons +
    process.dimuonsGlobal +
    process.dimuonsOneStandAloneMuon +
    process.dimuonsFilter
    )

process.dimuonsOneTrackPath = cms.Path(
    process.dimuonsHLTFilter +
    process.goodMuonRecoForDimuon +
    process.dimuonsOneTrack +
    process.dimuonsOneTrackFilter
    )

process.dimuonsMCTruth = cms.Path(
    process.dimuonsHLTFilter +
    process.mcTruthForDimuons
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

