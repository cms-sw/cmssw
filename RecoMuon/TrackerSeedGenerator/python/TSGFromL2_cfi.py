# The following comments couldn't be translated into the new config version:

#ServiceParameters

import FWCore.ParameterSet.Config as cms

from RecoMuon.GlobalTrackingTools.MuonTrackingRegionCommon_cff import *
from RecoMuon.TrackingTools.MuonServiceProxy_cff import *
from RecoMuon.TrackingTools.MuonErrorMatrixValues_cff import *
from RecoMuon.TrackerSeedGenerator.TrackerSeedCleaner_cff import *
hltL3TrajectorySeedFromL2 = cms.EDFilter("TSGFromL2Muon",
    MuonTrackingRegionCommon,
    MuonServiceProxy,
    tkSeedGenerator = cms.string('TSGFromCombinedHits'),
    TSGFromCombinedHits = cms.PSet(
        firstTSG = cms.PSet(
            ComponentName = cms.string('TSGFromOrderedHits'),
            OrderedHitsFactoryPSet = cms.PSet(
                ComponentName = cms.string('StandardHitTripletGenerator'),
                SeedingLayers = cms.string('PixelLayerTriplets'),
                GeneratorPSet = cms.PSet(
                    useBending = cms.bool(True),
                    useFixedPreFiltering = cms.bool(False),
                    ComponentName = cms.string('PixelTripletHLTGenerator'),
                    extraHitRPhitolerance = cms.double(0.06),
                    useMultScattering = cms.bool(True),
                    phiPreFiltering = cms.double(0.3),
                    extraHitRZtolerance = cms.double(0.06)
                )
            ),
            TTRHBuilder = cms.string('WithTrackAngle')
        ),
        ComponentName = cms.string('CombinedTSG'),
        thirdTSG = cms.PSet(
            PSetNames = cms.vstring('endcapTSG', 'barrelTSG'),
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
        PSetNames = cms.vstring('firstTSG', 'secondTSG', 'thirdTSG')
    ),
    TSGFromPropagation = cms.PSet(
        ErrorRescaling = cms.double(3.0),
        ComponentName = cms.string('TSGFromPropagation'),
        UpdateState = cms.bool(False),
        UseSecondMeasurements = cms.bool(False),
        MaxChi2 = cms.double(30.0),
        UseVertexState = cms.bool(True),
        Propagator = cms.string('SmartPropagatorAnyOpposite')
    ),
    UseTFileService = cms.untracked.bool(False),
    MuonCollectionLabel = cms.InputTag("hltL2Muons","UpdatedAtVtx"),
    TSGFromPixelTriplets = cms.PSet(
        ComponentName = cms.string('TSGFromOrderedHits'),
        OrderedHitsFactoryPSet = cms.PSet(
            ComponentName = cms.string('StandardHitTripletGenerator'),
            SeedingLayers = cms.string('PixelLayerTriplets'),
            GeneratorPSet = cms.PSet(
                useBending = cms.bool(True),
                useFixedPreFiltering = cms.bool(False),
                ComponentName = cms.string('PixelTripletHLTGenerator'),
                extraHitRPhitolerance = cms.double(0.06),
                useMultScattering = cms.bool(True),
                phiPreFiltering = cms.double(0.3),
                extraHitRZtolerance = cms.double(0.06)
            )
        ),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),
    TSGFromMixedPairs = cms.PSet(
        ComponentName = cms.string('TSGFromOrderedHits'),
        OrderedHitsFactoryPSet = cms.PSet(
            ComponentName = cms.string('StandardHitPairGenerator'),
            SeedingLayers = cms.string('MixedLayerPairs')
        ),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),
    #using TrackerSeedCleanerCommon
    TrackerSeedCleaner = cms.PSet(

    ),
    #this should not exist there !
    PtCut = cms.double(1.0),
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
    TSGFromPixelPairs = cms.PSet(
        ComponentName = cms.string('TSGFromOrderedHits'),
        OrderedHitsFactoryPSet = cms.PSet(
            ComponentName = cms.string('StandardHitPairGenerator'),
            SeedingLayers = cms.string('PixelLayerPairs')
        ),
        TTRHBuilder = cms.string('WithTrackAngle')
    )
)


