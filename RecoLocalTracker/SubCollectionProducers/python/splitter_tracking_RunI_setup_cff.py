import FWCore.ParameterSet.Config as cms
from RecoLocalTracker.SubCollectionProducers.SeedClusterRemover_cfi import *
from RecoTracker.IterativeTracking.ElectronSeeds_cff  import *
from RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi import *
from RecoTracker.MeasurementDet.MeasurementTrackerEventProducer_cfi import *#new in 700pre8
from RecoTracker.IterativeTracking.RunI_InitialStep_cff import *
from RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi import *
from RecoTracker.IterativeTracking.RunI_MixedTripletStep_cff import *
from RecoTracker.IterativeTracking.RunI_LowPtTripletStep_cff import *
from RecoTracker.IterativeTracking.RunI_PixelPairStep_cff import *
from RecoTracker.IterativeTracking.RunI_DetachedTripletStep_cff import *
from RecoTracker.IterativeTracking.RunI_PixelLessStep_cff import *
from RecoTracker.IterativeTracking.RunI_TobTecStep_cff import *
from RecoTracker.ConversionSeedGenerators.ConversionStep_cff import *

def customizeTracking(newpixclusters, newstripclusters, newpixrechits, newstriprechits):

    seedClusterRemover.stripClusters = newstripclusters
    seedClusterRemover.pixelClusters = newpixelclusters


    initialStepSeedClusterMask.stripClusters = newstripclusters
    initialStepSeedClusterMask.pixelClusters = newpixelclusters

    pixelPairStepSeedClusterMask.stripClusters  = newstripclusters
    pixelPairStepSeedClusterMask.pixelClusters = newpixelclusters
    
    mixedTripletStepSeedClusterMask.stripClusters  = newstripclusters
    mixedTripletStepSeedClusterMask.pixelClusters  = newpixelclusters
    
    pixelLessStepSeedClusterMask.stripClusters  = newstripclusters
    pixelLessStepSeedClusterMask.pixelClusters  = newpixelclusters

    tripletElectronClusterMask.stripClusters  =newstripclusters
    tripletElectronClusterMask.pixelClusters = newpixelclusters
    
    tripletElectronSeedLayers.BPix.HitProducer = newpixrechits
    tripletElectronSeedLayers.FPix.HitProducer = newpixrechits

    pixelPairElectronSeedLayers.BPix.HitProducer = newpixrechits
    pixelPairElectronSeedLayers.FPix.HitProducer = newpixrechits
    
    stripPairElectronSeedLayers.TIB.matchedRecHits = cms.InputTag(newstriprechits,"matchedRecHit")
    stripPairElectronSeedLayers.TID.matchedRecHits = cms.InputTag(newstriprechits,"matchedRecHit")
    stripPairElectronSeedLayers.TEC.matchedRecHits = cms.InputTag(newstriprechits,"matchedRecHit")

    
    MeasurementTracker.pixelClusterProducer = newpixelclusters
    MeasurementTracker.stripClusterProducer = newstripclusters

    MeasurementTrackerEvent.pixelClusterProducer =newpixelclusters
    MeasurementTrackerEvent.stripClusterProducer =newstripclusters
    

    initialStepSeedLayers.BPix.HitProducer = newpixrechits
    initialStepSeedLayers.FPix.HitProducer = newpixrechits
    
    initialStepSeeds.ClusterCheckPSet.PixelClusterCollectionLabel = newpixelclusters
    initialStepSeeds.ClusterCheckPSet.ClusterCollectionLabel      = newstripclusters
    

    lowPtTripletStepClusters.pixelClusters = newpixelclusters
    lowPtTripletStepClusters.stripClusters = newstripclusters
    lowPtTripletStepSeedLayers.BPix.HitProducer = newpixrechits
    lowPtTripletStepSeedLayers.FPix.HitProducer = newpixrechits
    lowPtTripletStepSeeds.ClusterCheckPSet.PixelClusterCollectionLabel = newpixelclusters
    lowPtTripletStepSeeds.ClusterCheckPSet.ClusterCollectionLabel      = newstripclusters

    pixelPairStepClusters.pixelClusters = newpixclusters
    pixelPairStepClusters.stripClusters = newstripclusters
    pixelPairStepSeedLayers.BPix.HitProducer =newpixrechits
    pixelPairStepSeedLayers.FPix.HitProducer =newpixrechits
    pixelPairStepSeeds.ClusterCheckPSet.PixelClusterCollectionLabel = newpixclusters
    pixelPairStepSeeds.ClusterCheckPSet.ClusterCollectionLabel      = newstripclusters

    detachedTripletStepClusters.pixelClusters = newpixclusters
    detachedTripletStepClusters.stripClusters = newstripclusters
    detachedTripletStepSeedLayers.BPix.HitProducer =newpixrechits
    detachedTripletStepSeedLayers.FPix.HitProducer =newpixrechits
    detachedTripletStepSeeds.ClusterCheckPSet.PixelClusterCollectionLabel = newpixclusters
    detachedTripletStepSeeds.ClusterCheckPSet.ClusterCollectionLabel      = newstripclusters
    detachedTripletStepClusters.stripRecHits =newstriprechits

    mixedTripletStepClusters.pixelClusters = newpixclusters
    mixedTripletStepClusters.stripClusters = newstripclusters
    mixedTripletStepSeedLayersA.BPix.HitProducer =newpixrechits
    mixedTripletStepSeedLayersA.FPix.HitProducer =newpixrechits
    mixedTripletStepSeedLayersB.BPix.HitProducer =newpixrechits
    mixedTripletStepSeedsA.ClusterCheckPSet.PixelClusterCollectionLabel = newpixclusters
    mixedTripletStepSeedsA.ClusterCheckPSet.ClusterCollectionLabel      = newstripclusters
    mixedTripletStepSeedsB.ClusterCheckPSet.PixelClusterCollectionLabel = newpixclusters
    mixedTripletStepSeedsB.ClusterCheckPSet.ClusterCollectionLabel      = newstripclusters
 

    pixelLessStepClusters.pixelClusters = newpixclusters
    pixelLessStepClusters.stripClusters = newstripclusters

    pixelLessStepSeeds.ClusterCheckPSet.PixelClusterCollectionLabel = newpixclusters
    pixelLessStepSeeds.ClusterCheckPSet.ClusterCollectionLabel      = newstripclusters

    tobTecStepClusters.pixelClusters = newpixclusters
    tobTecStepClusters.stripClusters = newstripclusters
    tobTecStepSeeds.ClusterCheckPSet.PixelClusterCollectionLabel = newpixclusters
    tobTecStepSeeds.ClusterCheckPSet.ClusterCollectionLabel      = newstripclusters
    mixedTripletStepSeedLayersA.TEC.matchedRecHits = cms.InputTag(newstriprechits,"matchedRecHit")
    mixedTripletStepSeedLayersB.TIB.matchedRecHits = cms.InputTag(newstriprechits,"matchedRecHit")

    pixelLessStepSeedLayers.TIB.matchedRecHits = cms.InputTag(newstriprechits,"matchedRecHit")
    pixelLessStepSeedLayers.TID.matchedRecHits = cms.InputTag(newstriprechits,"matchedRecHit")
    pixelLessStepSeedLayers.TEC.matchedRecHits = cms.InputTag(newstriprechits,"matchedRecHit")
    tobTecStepSeedLayers.TOB.matchedRecHits = cms.InputTag(newstriprechits,"matchedRecHit")
    tobTecStepSeedLayers.TEC.matchedRecHits = cms.InputTag(newstriprechits,"matchedRecHit")


    photonConvTrajSeedFromSingleLeg.ClusterCheckPSet.PixelClusterCollectionLabel = newpixclusters
    photonConvTrajSeedFromSingleLeg.ClusterCheckPSet.ClusterCollectionLabel      = newstripclusters
    convClusters.pixelClusters =newpixclusters 
    convClusters.stripClusters =newstripclusters
    convLayerPairs.BPix.HitProducer = newpixelrechits
    convLayerPairs.FPix.HitProducer = newpixelrechits
    convLayerPairs.TIB1.matchedRecHits = cms.InputTag(newstriprechits,"matchedRecHit")
    convLayerPairs.TIB2.matchedRecHits = cms.InputTag(newstriprechits,"matchedRecHit")
    convLayerPairs.TIB3.rphiRecHits = cms.InputTag(newstriprechits,"rphiRecHit")
    convLayerPairs.TIB4.rphiRecHits = cms.InputTag(newstriprechits,"rphiRecHit")
    convLayerPairs.TID1.matchedRecHits = cms.InputTag(newstriprechits,"matchedRecHit")
    convLayerPairs.TID1.stereoRecHits = cms.InputTag(newstriprechits,"stereoRecHitUnmatched")
    convLayerPairs.TID1.rphiRecHits = cms.InputTag(newstriprechits,"rphiRecHitUnmatched")
    convLayerPairs.TID2.matchedRecHits = cms.InputTag(newstriprechits,"matchedRecHit")
    convLayerPairs.TID2.stereoRecHits = cms.InputTag(newstriprechits,"stereoRecHitUnmatched")
    convLayerPairs.TID2.rphiRecHits = cms.InputTag(newstriprechits,"rphiRecHitUnmatched")
    convLayerPairs.TID3.matchedRecHits = cms.InputTag(newstriprechits,"matchedRecHit")
    convLayerPairs.TID3.stereoRecHits = cms.InputTag(newstriprechits,"stereoRecHitUnmatched")
    convLayerPairs.TID3.rphiRecHits = cms.InputTag(newstriprechits,"rphiRecHitUnmatched")
    convLayerPairs.TEC.matchedRecHits = cms.InputTag(newstriprechits,"matchedRecHit")
    convLayerPairs.TEC.stereoRecHits = cms.InputTag(newstriprechits,"stereoRecHitUnmatched")
    convLayerPairs.TEC.rphiRecHits = cms.InputTag(newstriprechits,"rphiRecHitUnmatched")
    convLayerPairs.TOB1.matchedRecHits = cms.InputTag(newstriprechits,"matchedRecHit")
    convLayerPairs.TOB2.matchedRecHits = cms.InputTag(newstriprechits,"matchedRecHit")
    convLayerPairs.TOB3.rphiRecHits = cms.InputTag(newstriprechits,"rphiRecHit")
    convLayerPairs.TOB4.rphiRecHits = cms.InputTag(newstriprechits,"rphiRecHit")
    convLayerPairs.TOB5.rphiRecHits = cms.InputTag(newstriprechits,"rphiRecHit")
    convLayerPairs.TOB6.rphiRecHits = cms.InputTag(newstriprechits,"rphiRecHit")
