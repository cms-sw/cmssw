import FWCore.ParameterSet.Config as cms
from RecoLocalTracker.SubCollectionProducers.SeedClusterRemover_cfi import *
from RecoTracker.IterativeTracking.InitialStep_cff import *
from RecoTracker.IterativeTracking.ElectronSeeds_cff  import *
from RecoTracker.IterativeTracking.LowPtTripletStep_cff import *
from RecoTracker.IterativeTracking.PixelPairStep_cff import *
from RecoTracker.IterativeTracking.DetachedTripletStep_cff import *
from RecoTracker.IterativeTracking.TobTecStep_cff import *
from RecoTracker.IterativeTracking.MixedTripletStep_cff import *
from RecoTracker.MeasurementDet.MeasurementTrackerEventProducer_cfi import *#new in 700pre8
from RecoTracker.IterativeTracking.PixelLessStep_cff import *
from  RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff import *
from RecoTracker.ConversionSeedGenerators.ConversionStep_cff import *


def customizeTracking(newpixclusters, newstripclusters, newpixrechits, newstriprechits):

    matchedsplitSiStripRecHits = cms.InputTag(newstriprechits,"matchedRecHit"), 
    matchedsplitSiPixelRecHits = cms.InputTag(newpixrechits,"matchedRecHit"), 
    rphisplitSiStripRecHits = cms.InputTag(newstriprechits,"rphiRecHit"), 
    stereosplitSiStripRecHits = cms.InputTag(newpixrechits,"stereoRecHit"), 
    
    seedClusterRemover.stripClusters = newstripclusters
    seedClusterRemover.pixelClusters = newpixclusters



    initialStepSeedLayers.BPix.HitProducer =newpixrechits
    initialStepSeedLayers.FPix.HitProducer =newpixrechits
    initialStepClusters.pixelClusters = newpixclusters
    initialStepClusters.stripClusters = newstripclusters
    initialStepClusters.stripRecHits  =newstriprechits
    initialStepSeeds.ClusterCheckPSet.PixelClusterCollectionLabel = newpixclusters
    initialStepSeeds.ClusterCheckPSet.ClusterCollectionLabel      = newstripclusters
    #initialStepSeeds.OrderedHitsFactoryPSet.SeedingLayers = 'initialStepSeedLayers'

    initialStepSeedClusterMask.stripClusters = newstripclusters
    initialStepSeedClusterMask.pixelClusters = newpixclusters
    
    pixelPairStepSeedClusterMask.stripClusters  = newstripclusters
    pixelPairStepSeedClusterMask.pixelClusters = newpixclusters
    
    mixedTripletStepSeedClusterMask.stripClusters  = newstripclusters
    mixedTripletStepSeedClusterMask.pixelClusters = newpixclusters

    
    tripletElectronClusterMask.stripClusters  = newstripclusters
    tripletElectronClusterMask.pixelClusters = newpixclusters

    tripletElectronSeedLayers.BPix.HitProducer =newpixrechits
    tripletElectronSeedLayers.FPix.HitProducer =newpixrechits
    
    pixelPairElectronSeedLayers.BPix.HitProducer =newpixrechits
    pixelPairElectronSeedLayers.FPix.HitProducer =newpixrechits
    
    stripPairElectronSeedLayers.TIB.matchedRecHits = cms.InputTag(newstriprechits,"matchedRecHit")
    stripPairElectronSeedLayers.TID.matchedRecHits = cms.InputTag(newstriprechits,"matchedRecHit")	
    stripPairElectronSeedLayers.TEC.matchedRecHits = cms.InputTag(newstriprechits,"matchedRecHit")



    lowPtTripletStepClusters.pixelClusters = newpixclusters
    lowPtTripletStepClusters.stripClusters = newstripclusters
    lowPtTripletStepSeedLayers.BPix.HitProducer =newpixrechits
    lowPtTripletStepSeedLayers.FPix.HitProducer =newpixrechits
    lowPtTripletStepSeeds.ClusterCheckPSet.PixelClusterCollectionLabel = newpixclusters
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
    mixedTripletStepSeedLayersA.TEC.matchedRecHits = cms.InputTag(newstriprechits,"matchedRecHit")
    mixedTripletStepSeedLayersB.TIB.matchedRecHits = cms.InputTag(newstriprechits,"matchedRecHit")



    pixelLessStepSeedClusterMask.stripClusters  = newstripclusters
    pixelLessStepSeedClusterMask.pixelClusters = newpixclusters
    pixelLessStepClusters.pixelClusters = newpixclusters
    pixelLessStepClusters.stripClusters = newstripclusters
    pixelLessStepSeeds.ClusterCheckPSet.PixelClusterCollectionLabel = newpixclusters
    pixelLessStepSeeds.ClusterCheckPSet.ClusterCollectionLabel      = newstripclusters
    pixelLessStepSeedClusters.stripRecHits = newstriprechits
    pixelLessStepSeedClusters.pixelClusters = newpixclusters
    pixelLessStepSeedClusters.stripClusters = newstripclusters
    pixelLessStepSeedLayers.MTIB.rphiRecHits = cms.InputTag(newstriprechits,"rphiRecHit")
    pixelLessStepSeedLayers.MTID.rphiRecHits = cms.InputTag(newstriprechits,"rphiRecHit")
    pixelLessStepSeedLayers.MTEC.rphiRecHits = cms.InputTag(newstriprechits,"rphiRecHit")
    
    pixelLessStepSeedLayers.TIB.matchedRecHits = cms.InputTag(newstriprechits,"matchedRecHit")
    pixelLessStepSeedLayers.TID.matchedRecHits = cms.InputTag(newstriprechits,"matchedRecHit")
    pixelLessStepSeedLayers.TEC.matchedRecHits = cms.InputTag(newstriprechits,"matchedRecHit")
    
    tobTecStepClusters.pixelClusters = newpixclusters
    tobTecStepClusters.stripClusters = newstripclusters
    tobTecStepSeedClusters.pixelClusters = newpixclusters
    tobTecStepSeedClusters.stripClusters = newstripclusters
    tobTecStepSeedClusters.stripRecHits =newstriprechits
    tobTecStepSeedLayersPair.TOB.matchedRecHits  = cms.InputTag(newstriprechits,"matchedRecHit")
    tobTecStepSeedLayersPair.TEC.matchedRecHits  = cms.InputTag(newstriprechits,"matchedRecHit")
    tobTecStepSeedLayersTripl.TOB.matchedRecHits = cms.InputTag(newstriprechits,"matchedRecHit")
    tobTecStepSeedLayersTripl.MTOB.rphiRecHits   = cms.InputTag(newstriprechits,"rphiRecHit")
    tobTecStepSeedLayersTripl.MTEC.rphiRecHits   = cms.InputTag(newstriprechits,"rphiRecHit")


    MeasurementTrackerEvent.pixelClusterProducer = newpixclusters
    MeasurementTrackerEvent.stripClusterProducer = newstripclusters

    globalSeedsFromTriplets.ClusterCheckPSet.ClusterCollectionLabel=newstripclusters
    globalSeedsFromTriplets.ClusterCheckPSet.PixelClusterCollectionLabel=newpixclusters



    photonConvTrajSeedFromSingleLeg.ClusterCheckPSet.PixelClusterCollectionLabel = newpixclusters
    photonConvTrajSeedFromSingleLeg.ClusterCheckPSet.ClusterCollectionLabel      = newstripclusters
    convClusters.pixelClusters = newpixclusters
    convClusters.stripClusters = newstripclusters
    convLayerPairs.BPix.HitProducer =newpixrechits
    convLayerPairs.FPix.HitProducer =newpixrechits
    convLayerPairs.TIB1.matchedRecHits = cms.InputTag(newstriprechits,"matchedRecHit")
    convLayerPairs.TIB2.matchedRecHits = cms.InputTag(newstriprechits,"matchedRecHit")
    convLayerPairs.TID1.matchedRecHits = cms.InputTag(newstriprechits,"matchedRecHit")
    convLayerPairs.TID2.matchedRecHits = cms.InputTag(newstriprechits,"matchedRecHit")
    convLayerPairs.TID3.matchedRecHits = cms.InputTag(newstriprechits,"matchedRecHit")
    convLayerPairs.TEC.matchedRecHits  = cms.InputTag(newstriprechits,"matchedRecHit")
    convLayerPairs.TOB1.matchedRecHits = cms.InputTag(newstriprechits,"matchedRecHit")
    convLayerPairs.TOB2.matchedRecHits = cms.InputTag(newstriprechits,"matchedRecHit")
    convLayerPairs.TIB3.rphiRecHits = cms.InputTag(newstriprechits,"rphiRecHit")
    convLayerPairs.TIB4.rphiRecHits = cms.InputTag(newstriprechits,"rphiRecHit")
    convLayerPairs.TID1.rphiRecHits = cms.InputTag(newstriprechits,"rphiRecHit")
    convLayerPairs.TID2.rphiRecHits = cms.InputTag(newstriprechits,"rphiRecHit")
    convLayerPairs.TID3.rphiRecHits = cms.InputTag(newstriprechits,"rphiRecHit")
    convLayerPairs.TEC.rphiRecHits  = cms.InputTag(newstriprechits,"rphiRecHit")
    convLayerPairs.TOB3.rphiRecHits = cms.InputTag(newstriprechits,"rphiRecHit")
    convLayerPairs.TOB4.rphiRecHits = cms.InputTag(newstriprechits,"rphiRecHit")
    convLayerPairs.TOB5.rphiRecHits = cms.InputTag(newstriprechits,"rphiRecHit")
    convLayerPairs.TOB6.rphiRecHits = cms.InputTag(newstriprechits,"rstrophiRecHit")
    convLayerPairs.TID1.stereoRecHits = cms.InputTag(newstriprechits,"stereoRecHit")
    convLayerPairs.TID2.stereoRecHits = cms.InputTag(newstriprechits,"stereoRecHit")
    convLayerPairs.TID3.stereoRecHits = cms.InputTag(newstriprechits,"stereoRecHit")
    convLayerPairs.TEC.stereoRecHits  = cms.InputTag(newstriprechits,"stereoRecHit")
