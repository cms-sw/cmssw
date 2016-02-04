import FWCore.ParameterSet.Config as cms

#Includes for tracking
from RecoTracker.Configuration.RecoTracker_cff import *
from RecoLocalTracker.Configuration.RecoLocalTracker_cff import *

#First Step
from RecoTracker.IterativeTracking.FirstStep_cff import *
newClusters.pixelClusters = cms.InputTag('siClusterTranslator')
newClusters.stripClusters = cms.InputTag('siClusterTranslator')
newStripRecHits.StripCPE = cms.string('FastStripCPE')
newPixelRecHits.CPE = cms.string('FastPixelCPE')
newMeasurementTracker.StripCPE = cms.string('FastStripCPE')
newMeasurementTracker.PixelCPE = cms.string('FastPixelCPE')

#Second Step
from RecoTracker.IterativeTracking.SecStep_cff import *
secPixelRecHits.CPE = cms.string('FastPixelCPE')
secStripRecHits.StripCPE = cms.string('FastStripCPE')
secMeasurementTracker.StripCPE = cms.string('FastStripCPE')
secMeasurementTracker.PixelCPE = cms.string('FastPixelCPE')

#Third Step
from RecoTracker.IterativeTracking.ThStep_cff import *
thPixelRecHits.CPE = cms.string('FastPixelCPE')
thStripRecHits.StripCPE = cms.string('FastStripCPE')
thMeasurementTracker.StripCPE = cms.string('FastStripCPE')
thMeasurementTracker.PixelCPE = cms.string('FastPixelCPE')

#Fourth Step
from RecoTracker.IterativeTracking.PixelLessStep_cff import *
fourthPixelRecHits.CPE = cms.string('FastPixelCPE')
fourthStripRecHits.StripCPE = cms.string('FastStripCPE')
fourthMeasurementTracker.StripCPE = cms.string('FastStripCPE')
fourthMeasurementTracker.PixelCPE = cms.string('FastPixelCPE')

#Fifth Step
from RecoTracker.IterativeTracking.TobTecStep_cff import *
fifthPixelRecHits.CPE = cms.string('FastPixelCPE')
fifthStripRecHits.StripCPE = cms.string('FastStripCPE')
fifthMeasurementTracker.StripCPE = cms.string('FastStripCPE')
fifthMeasurementTracker.PixelCPE = cms.string('FastPixelCPE')

#Strips
import RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi
siStripMatchedRecHits.StripCPE = cms.string('FastStripCPE')
siStripMatchedRecHits.ClusterProducer = cms.string('siClusterTranslator')

#Pixels
from RecoLocalTracker.SiPixelRecHits.PixelCPEESProducers_cff import *
import RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi
siPixelRecHits.src = cms.InputTag('siClusterTranslator')
siPixelRecHits.CPE = cms.string('FastPixelCPE')
from RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff import *
globalSeedsFromTriplets.TTRHBuilder = cms.string("FastPixelCPE")

#Transient Rec Hits
import RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi
ttrhbwr.StripCPE = cms.string('FastStripCPE')
ttrhbwr.PixelCPE = cms.string('FastPixelCPE')
import RecoTracker.TransientTrackingRecHit.TTRHBuilderWithTemplate_cfi
TTRHBuilderAngleAndTemplate.StripCPE = cms.string('FastStripCPE')
TTRHBuilderAngleAndTemplate.PixelCPE = cms.string('FastPixelCPE')
import RecoTracker.TkSeedingLayers.TTRHBuilderWithoutAngle4PixelPairs_cfi
myTTRHBuilderWithoutAngle4PixelPairs.PixelCPE = cms.string("FastPixelCPE")
import RecoTracker.TkSeedingLayers.TTRHBuilderWithoutAngle4PixelTriplets_cfi
myTTRHBuilderWithoutAngle4PixelTriplets.PixelCPE = cms.string("FastPixelCPE")
import RecoTracker.TkSeedingLayers.TTRHBuilderWithoutAngle4MixedPairs_cfi
myTTRHBuilderWithoutAngle4MixedPairs.PixelCPE = cms.string("FastPixelCPE")
import RecoTracker.TkSeedingLayers.TTRHBuilderWithoutAngle4MixedTriplets_cfi
myTTRHBuilderWithoutAngle4MixedTriplets.PixelCPE = cms.string("FastPixelCPE")
import RecoTracker.TkSeedingLayers.MixedLayerPairs_cfi

#Tracks
import RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi
MeasurementTracker.stripClusterProducer = cms.string('siClusterTranslator')
MeasurementTracker.pixelClusterProducer = cms.string('siClusterTranslator')
MeasurementTracker.StripCPE = cms.string('FastStripCPE')
MeasurementTracker.PixelCPE = cms.string('FastPixelCPE')

#Making sure not to use the Seed Comparitor
##from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import* 
newSeedFromTriplets.SeedComparitorPSet.ComponentName = 'none'
secTriplets.SeedComparitorPSet.ComponentName = 'none'


siClusterTranslator = cms.EDProducer("SiClusterTranslator")

translationAndTracking_wodEdx = cms.Sequence(siClusterTranslator*siPixelRecHits*siStripMatchedRecHits*iterTracking*trackCollectionMerging*newCombinedSeeds)
translationAndTracking = cms.Sequence(siClusterTranslator*siPixelRecHits*siStripMatchedRecHits*ckftracks)
