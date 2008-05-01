import FWCore.ParameterSet.Config as cms

from RecoPixelVertexing.PixelTriplets.PixelTripletHLTGenerator_cfi import *
from RecoMuon.TrackingTools.MuonErrorMatrixValues_cff import *
TSGsBlock = cms.PSet(
    TSGFromCombinedHits = cms.PSet(
        firstTSG = cms.PSet(
            ComponentName = cms.string('TSGFromOrderedHits'),
            OrderedHitsFactoryPSet = cms.PSet(
                ComponentName = cms.string('StandardHitTripletGenerator'),
                SeedingLayers = cms.string('PixelLayerTriplets'),
                GeneratorPSet = cms.PSet(
                    PixelTripletHLTGenerator
                )
            ),
            TTRHBuilder = cms.string('WithTrackAngle')
        ),
        ComponentName = cms.string('CombinedTSG'),
        thirdTSG = cms.PSet(
            PSetNames = cms.vstring('endcapTSG', 
                'barrelTSG'),
            ComponentName = cms.string('DualByEtaTSG'),
            endcapTSG = cms.PSet(
                ComponentName = cms.string('TSGFromOrderedHits'),
                OrderedHitsFactoryPSet = cms.PSet(
                    ComponentName = cms.string('StandardHitPairGenerator'),
                    SeedingLayers = cms.string('MixedLayerPairs')
                ),
                TTRHBuilder = cms.string('WithTrackAngle')
            ),
            etaSeparation = cms.double(2.0),
            barrelTSG = cms.PSet(

            )
        ),
        secondTSG = cms.PSet(
            ComponentName = cms.string('TSGFromOrderedHits'),
            OrderedHitsFactoryPSet = cms.PSet(
                ComponentName = cms.string('StandardHitPairGenerator'),
                SeedingLayers = cms.string('PixelLayerPairs')
            ),
            TTRHBuilder = cms.string('WithTrackAngle')
        ),
        PSetNames = cms.vstring('firstTSG', 
            'secondTSG')
    ),
    TSGFromPropagation = cms.PSet(
        ErrorRescaling = cms.double(3.0),
        ComponentName = cms.string('TSGFromPropagation'),
        errorMatrixPset = cms.PSet(

        ),
        UpdateState = cms.bool(False),
        UseSecondMeasurements = cms.bool(False),
        MaxChi2 = cms.double(30.0),
        UseVertexState = cms.bool(True),
        Propagator = cms.string('SmartPropagatorAnyOpposite')
    ),
    TSGFromPixelTriplets = cms.PSet(
        ComponentName = cms.string('TSGFromOrderedHits'),
        OrderedHitsFactoryPSet = cms.PSet(
            ComponentName = cms.string('StandardHitTripletGenerator'),
            SeedingLayers = cms.string('PixelLayerTriplets'),
            GeneratorPSet = cms.PSet(
                PixelTripletHLTGenerator
            )
        ),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),
    TSGForRoadSearchOI = cms.PSet(
        propagatorCompatibleName = cms.string('SteppingHelixPropagatorAny'),
        option = cms.uint32(3),
        ComponentName = cms.string('TSGForRoadSearch'),
        errorMatrixPset = cms.PSet(
            MuonErrorMatrixValues,
            action = cms.string('use'),
            atIP = cms.bool(True)
        ),
        propagatorName = cms.string('SteppingHelixPropagatorAlong'),
        manySeeds = cms.bool(False),
        copyMuonRecHit = cms.bool(False),
        maxChi2 = cms.double(40.0)
    ),
    TSGFromMixedPairs = cms.PSet(
        ComponentName = cms.string('TSGFromOrderedHits'),
        OrderedHitsFactoryPSet = cms.PSet(
            ComponentName = cms.string('StandardHitPairGenerator'),
            SeedingLayers = cms.string('MixedLayerPairs')
        ),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),
    TSGForRoadSearchIOpxl = cms.PSet(
        propagatorCompatibleName = cms.string('SteppingHelixPropagatorAny'),
        option = cms.uint32(4),
        ComponentName = cms.string('TSGForRoadSearch'),
        errorMatrixPset = cms.PSet(
            MuonErrorMatrixValues,
            action = cms.string('use'),
            atIP = cms.bool(True)
        ),
        propagatorName = cms.string('SteppingHelixPropagatorAlong'),
        manySeeds = cms.bool(False),
        copyMuonRecHit = cms.bool(False),
        maxChi2 = cms.double(40.0)
    ),
    TSGFromPixelPairs = cms.PSet(
        ComponentName = cms.string('TSGFromOrderedHits'),
        OrderedHitsFactoryPSet = cms.PSet(
            ComponentName = cms.string('StandardHitPairGenerator'),
            SeedingLayers = cms.string('PixelLayerPairs')
        ),
        TTRHBuilder = cms.string('WithTrackAngle')
    )
)

