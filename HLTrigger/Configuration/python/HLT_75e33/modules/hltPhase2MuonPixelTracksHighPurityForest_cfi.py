import FWCore.ParameterSet.Config as cms

# High-purity selection of the muon pixel tracks (pixel-track chain) with a forest of boosted decision trees
hltPhase2MuonPixelTracksHighPurityForest = cms.EDProducer('MuonIOTracksForestSelector',
    tracks = cms.InputTag('hltPhase2MuonPixelTracks'),
    l1TkMuons = cms.InputTag("l1tTkMuonsGmt"),
    dtSegments = cms.InputTag("hltDt4DSegments"),
    cscSegments = cms.InputTag("hltCscSegments"),
    gemSegments = cms.InputTag("hltGemSegments"),
    gemRecHits = cms.InputTag("hltGemRecHits"),
    rpcRecHits = cms.InputTag("hltRpcRecHits"),
    model = cms.FileInPath('RecoMuon/L3TrackFinder/data/IO/muonHP_IO_pixelPath_forest_v1.bin'),
    decisionThreshold = cms.double(0.7648054361343384),
    ptBinEdges = cms.vdouble(0.0, 3.0, 5.0, 10.0, 50.0, 200.0),
    decisionThresholds = cms.vdouble(
        0.26123544573783875,  # pT [0, 3)
        0.5858924984931946,  # pT [3, 5)
        0.9377449750900269,  # pT [5, 10)
        0.9989855289459229,  # pT [10, 50)
        0.9998182058334351,  # pT [50, 200)
        0.9997901320457458,  # pT [200, inf)
    ),
    nFeatures = cms.int32(46),
    dumpFeatures = cms.untracked.bool(False),
)
