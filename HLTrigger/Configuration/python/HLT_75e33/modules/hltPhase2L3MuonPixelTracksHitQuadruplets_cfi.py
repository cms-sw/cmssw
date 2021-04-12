import FWCore.ParameterSet.Config as cms

hltPhase2L3MuonPixelTracksHitQuadruplets = cms.EDProducer("CAHitQuadrupletEDProducer",
    CAHardPtCut = cms.double(0),
    CAOnlyOneLastHitPerLayerFilter = cms.optional.bool,
    CAPhiCut = cms.double(0.2),
    CAThetaCut = cms.double(0.0012),
    SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('LowPtClusterShapeSeedComparitor'),
        clusterShapeCacheSrc = cms.InputTag("siPixelClusterShapeCache"),
        clusterShapeHitFilter = cms.string('ClusterShapeHitFilter')
    ),
    doublets = cms.InputTag("hltPhase2L3MuonPixelTracksHitDoublets"),
    extraHitRPhitolerance = cms.double(0.032),
    fitFastCircle = cms.bool(True),
    fitFastCircleChi2Cut = cms.bool(True),
    maxChi2 = cms.PSet(
        enabled = cms.bool(True),
        pt1 = cms.double(0.7),
        pt2 = cms.double(2.0),
        value1 = cms.double(200.0),
        value2 = cms.double(50.0)
    ),
    mightGet = cms.optional.untracked.vstring,
    useBendingCorrection = cms.bool(True)
)
