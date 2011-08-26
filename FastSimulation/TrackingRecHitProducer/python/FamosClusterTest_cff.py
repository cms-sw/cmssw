import FWCore.ParameterSet.Config as cms

#Includes for tracking
from RecoTracker.Configuration.RecoTracker_cff import *
from RecoLocalTracker.Configuration.RecoLocalTracker_cff import *

#First Step
from RecoTracker.IterativeTracking.LowPtTripletStep_cff import *
lowPtTripletStepClusters.pixelClusters = cms.InputTag('siClusterTranslator')
lowPtTripletStepClusters.stripClusters = cms.InputTag('siClusterTranslator')
#lowPtTripletStepMeasurementTracker.StripCPE = cms.string('FastStripCPE')
#lowPtTripletStepMeasurementTracker.PixelCPE = cms.string('FastPixelCPE')

#Second Step
from RecoTracker.IterativeTracking.PixelPairStep_cff import *
#pixelPairStepMeasurementTracker.StripCPE = cms.string('FastStripCPE')
#pixelPairStepMeasurementTracker.PixelCPE = cms.string('FastPixelCPE')


#Third Step
from RecoTracker.IterativeTracking.DetachedTripletStep_cff import *
#detachedTripletStepMeasurementTracker.StripCPE = cms.string('FastStripCPE')
#detachedTripletStepMeasurementTracker.PixelCPE = cms.string('FastPixelCPE')

#Fourth Step
from RecoTracker.IterativeTracking.PixelLessStep_cff import *
#pixelLessStepMeasurementTracker.StripCPE = cms.string('FastStripCPE')
#pixelLessStepMeasurementTracker.PixelCPE = cms.string('FastPixelCPE')

#Fifth Step
from RecoTracker.IterativeTracking.TobTecStep_cff import *
#tobTecStepMeasurementTracker.StripCPE = cms.string('FastStripCPE')
#tobTecStepMeasurementTracker.PixelCPE = cms.string('FastPixelCPE')

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
initialStepSeeds.SeedComparitorPSet.ComponentName = 'none'
lowPtTripletStepSeeds.SeedComparitorPSet.ComponentName = 'none'
detachedTripletStepSeeds.SeedComparitorPSet.ComponentName = 'none'

siClusterTranslator = cms.EDProducer("SiClusterTranslator")

translationAndTracking_wodEdx = cms.Sequence(siClusterTranslator*siPixelRecHits*siStripMatchedRecHits*ckftracks_wodEdX)
translationAndTracking = cms.Sequence(siClusterTranslator*siPixelRecHits*siStripMatchedRecHits*ckftracks)
