import FWCore.ParameterSet.Config as cms

PixelTripletHLTGeneratorWithFilter = cms.PSet(
    ComponentName = cms.string('PixelTripletHLTGenerator'),
    SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('LowPtClusterShapeSeedComparitor'),
        clusterShapeCacheSrc = cms.InputTag("siPixelClusterShapeCache"),
        clusterShapeHitFilter = cms.string('ClusterShapeHitFilter')
    ),
    extraHitRPhitolerance = cms.double(0.016),
    extraHitRZtolerance = cms.double(0.02),
    maxElement = cms.uint32(100000),
    phiPreFiltering = cms.double(0.3),
    useBending = cms.bool(True),
    useFixedPreFiltering = cms.bool(False),
    useMultScattering = cms.bool(True)
)