import FWCore.ParameterSet.Config as cms

# High-purity selection of the muon inside-out tracks (LST-seeded chain) with a forest of boosted decision trees
hltPhase2MuonIOTrackSelectionHighPurityForest = cms.EDProducer("MuonIOTracksForestSelector",
    tracks = cms.InputTag("hltPhase2MuonIOTracks"),
    l1TkMuons = cms.InputTag("l1tTkMuonsGmt"),
    dtSegments = cms.InputTag("hltDt4DSegments"),
    cscSegments = cms.InputTag("hltCscSegments"),
    gemSegments = cms.InputTag("hltGemSegments"),
    gemRecHits = cms.InputTag("hltGemRecHits"),
    rpcRecHits = cms.InputTag("hltRpcRecHits"),
    model = cms.FileInPath('RecoMuon/L3TrackFinder/data/IO/muonHP_IO_seedsPath_forest_v1.bin'),
    decisionThreshold = cms.double(0.591177761554718),
    ptBinEdges = cms.vdouble(0.0, 3.0, 5.0, 10.0, 50.0, 200.0),
    decisionThresholds = cms.vdouble(
        0.08881093561649323,  # pT [0, 3)
        0.4612923264503479,  # pT [3, 5)
        0.9276527166366577,  # pT [5, 10)
        0.9981892704963684,  # pT [10, 50)
        0.9987490773200989,  # pT [50, 200)
        0.999536395072937,  # pT [200, inf)
    ),
    nFeatures = cms.int32(46),
    dumpFeatures = cms.untracked.bool(False),
)
