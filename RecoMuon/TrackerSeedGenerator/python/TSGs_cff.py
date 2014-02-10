import FWCore.ParameterSet.Config as cms

from RecoPixelVertexing.PixelTriplets.PixelTripletHLTGenerator_cfi import *
from RecoMuon.TrackingTools.MuonErrorMatrixValues_cff import *
TSGsBlock = cms.PSet(
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
    ),
    TSGFromPropagation = cms.PSet(
      MuonErrorMatrixValues,
      ComponentName = cms.string( "TSGFromPropagation" ),
      Propagator = cms.string( "SmartPropagatorAnyOpposite" ),
      MaxChi2 = cms.double( 40.0 ),
      ResetMethod = cms.string("matrix"),
      ErrorRescaling = cms.double(3.0),
      SigmaZ = cms.double(25.0),
      UseVertexState = cms.bool( True ),
      UpdateState = cms.bool( True ),
      SelectState = cms.bool( False ),
      beamSpot = cms.InputTag("hltOfflineBeamSpot")
      ##       errorMatrixPset = cms.PSet(
      ##           MuonErrorMatrixValues,
      ##           action = cms.string('use'),
      ##           atIP = cms.bool(True)
      ##           ),
      #UseSecondMeasurements = cms.bool( False )
      ),
    TSGFromPixelTriplets = cms.PSet(
        ComponentName = cms.string('TSGFromOrderedHits'),
        OrderedHitsFactoryPSet = cms.PSet(
            ComponentName = cms.string('StandardHitTripletGenerator'),
            SeedingLayers = cms.InputTag('PixelLayerTriplets'),
            GeneratorPSet = cms.PSet(
                PixelTripletHLTGenerator
            )
        ),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),
    TSGForRoadSearchOI = cms.PSet(
        MuonErrorMatrixValues,
        propagatorCompatibleName = cms.string('SteppingHelixPropagatorOpposite'),
        option = cms.uint32(3),
        ComponentName = cms.string('TSGForRoadSearch'),
        ##         errorMatrixPset = cms.PSet(
        ##         MuonErrorMatrixValues,
        ##             action = cms.string('use'),
        ##             atIP = cms.bool(True)
        ##         ),
        propagatorName = cms.string('SteppingHelixPropagatorAlong'),
        manySeeds = cms.bool(False),
        copyMuonRecHit = cms.bool(False),
        maxChi2 = cms.double(40.0)
    ),
    TSGFromMixedPairs = cms.PSet(
        ComponentName = cms.string('TSGFromOrderedHits'),
        OrderedHitsFactoryPSet = cms.PSet(
            ComponentName = cms.string('StandardHitPairGenerator'),
            SeedingLayers = cms.InputTag('MixedLayerPairs'),
	    useOnDemandTracker = cms.untracked.int32( 0 ),
	    maxElement = cms.uint32( 0 )
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
            SeedingLayers = cms.InputTag('PixelLayerPairs'),
	    useOnDemandTracker = cms.untracked.int32( 0 ),
	    maxElement = cms.uint32( 0 )
        ),
        TTRHBuilder = cms.string('WithTrackAngle')
    )
)


