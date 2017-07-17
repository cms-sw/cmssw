import FWCore.ParameterSet.Config as cms

TSGFromCombinedHits = cms.PSet(
    ComponentName = cms.string('CombinedTSG'),
    PSetNames = cms.vstring('firstTSG','secondTSG'),
    
    firstTSG = cms.PSet(
    ComponentName = cms.string('TSGFromOrderedHits'),
    OrderedHitsFactoryPSet = cms.PSet(
    ComponentName = cms.string('StandardHitTripletGenerator'),
    SeedingLayers = cms.InputTag('PixelLayerTriplets'),
    GeneratorPSet = cms.PSet(
    useBending = cms.bool(True),
    useFixedPreFiltering = cms.bool(False),
    phiPreFiltering = cms.double(0.3),
    extraHitRPhitolerance = cms.double(0.06),
    useMultScattering = cms.bool(True),
    ComponentName = cms.string('PixelTripletHLTGenerator'),
    extraHitRZtolerance = cms.double(0.06),
    maxElement = cms.uint32( 10000 )
    )
    ),
    TTRHBuilder = cms.string('WithTrackAngle')
    ),
    
    secondTSG = cms.PSet(
    ComponentName = cms.string('TSGFromOrderedHits'),
    OrderedHitsFactoryPSet = cms.PSet(
    ComponentName = cms.string('StandardHitPairGenerator'),
    SeedingLayers = cms.InputTag('PixelLayerPairs'),
    useOnDemandTracker = cms.untracked.int32( 0 ),
    maxElement = cms.uint32( 0 )
    ),
    TTRHBuilder = cms.string('WithTrackAngle')
    ),
    thirdTSG = cms.PSet(
    ComponentName = cms.string('DualByEtaTSG'),
    PSetNames = cms.vstring('endcapTSG','barrelTSG'),
    barrelTSG = cms.PSet(    ),
    endcapTSG = cms.PSet(
    ComponentName = cms.string('TSGFromOrderedHits'),
    OrderedHitsFactoryPSet = cms.PSet(
    ComponentName = cms.string('StandardHitPairGenerator'),
    SeedingLayers = cms.InputTag('MixedLayerPairs'),
    useOnDemandTracker = cms.untracked.int32( 0 ),
    maxElement = cms.uint32( 0 )
    ),
    TTRHBuilder = cms.string('WithTrackAngle')
    ),
    etaSeparation = cms.double(2.0)
    )
    )
