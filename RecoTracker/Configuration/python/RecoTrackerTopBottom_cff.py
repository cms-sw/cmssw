import FWCore.ParameterSet.Config as cms
import copy

from RecoLocalTracker.SubCollectionProducers.ClusterSelectorTopBottom_cfi import *
from RecoLocalTracker.Configuration.RecoLocalTracker_Cosmics_cff import *
from RecoTracker.SpecialSeedGenerators.CombinatorialSeedGeneratorForCosmicsP5_cff import combinatorialcosmicseedfinderP5
from RecoTracker.SpecialSeedGenerators.SimpleCosmicBONSeeder_cfi import simpleCosmicBONSeeds
from RecoTracker.TkSeedGenerator.GlobalCombinedSeeds_cff import globalCombinedSeeds
from RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cff import MeasurementTracker
from RecoTracker.CkfPattern.CkfTrackCandidatesP5_cff import GroupedCkfTrajectoryBuilderP5
from RecoTracker.CkfPattern.CkfTrackCandidatesP5_cff import ckfTrackCandidatesP5
from RecoTracker.TrackProducer.CTFFinalFitWithMaterialP5_cff import ctfWithMaterialTracksCosmics
from RecoTracker.SpecialSeedGenerators.CosmicSeedP5Pairs_cff import cosmicseedfinderP5
from RecoTracker.SingleTrackPattern.CosmicTrackFinderP5_cff import cosmicCandidateFinderP5
from RecoTracker.SingleTrackPattern.CosmicTrackFinderP5_cff import cosmictrackfinderCosmics
from RecoTracker.RoadSearchSeedFinder.RoadSearchSeedsP5_cff import roadSearchSeedsP5
from RecoTracker.RoadSearchCloudMaker.RoadSearchCloudsP5_cff import roadSearchCloudsP5
from RecoTracker.RoadSearchTrackCandidateMaker.RoadSearchTrackCandidatesP5_cff import rsTrackCandidatesP5
from RecoTracker.TrackProducer.RSFinalFitWithMaterialP5_cff import rsWithMaterialTracksCosmics

siPixelRecHitsTop = siPixelRecHits.clone(src = cms.InputTag("siPixelClustersTop"))
siPixelRecHitsBottom = siPixelRecHits.clone(src = cms.InputTag("siPixelClustersBottom"))
siStripMatchedRecHitsTop = siStripMatchedRecHits.clone(ClusterProducer = cms.InputTag('siStripClustersTop'))
siStripMatchedRecHitsBottom = siStripMatchedRecHits.clone(ClusterProducer = cms.InputTag('siStripClustersBottom'))

from RecoLocalTracker.SubCollectionProducers.TopBottomClusterInfoProducer_cfi import topBottomClusterInfoProducer
topBottomClusterInfoProducerTop = topBottomClusterInfoProducer.clone()
topBottomClusterInfoProducerBottom = topBottomClusterInfoProducer.clone(
    stripClustersNew = cms.InputTag("siStripClustersBottom"),
    pixelClustersNew = cms.InputTag("siPixelClustersBottom"),
    stripMonoHitsNew = cms.InputTag("siStripMatchedRecHitsBottom","rphiRecHit"),
    stripStereoHitsNew = cms.InputTag("siStripMatchedRecHitsBottom","stereoRecHit"),
    pixelHitsNew = cms.InputTag("siPixelRecHitsBottom")
)
###LOCAL RECO SEQUENCE
trackerlocalrecoTop = cms.Sequence(((siPixelClustersTop*siPixelRecHitsTop)+(siStripClustersTop*siStripMatchedRecHitsTop))*topBottomClusterInfoProducerTop)
trackerlocalrecoBottom = cms.Sequence(((siPixelClustersBottom*siPixelRecHitsBottom)+(siStripClustersBottom*siStripMatchedRecHitsBottom))*topBottomClusterInfoProducerBottom)

###CKF TOP
combinatorialcosmicseedfinderP5Top = copy.deepcopy(combinatorialcosmicseedfinderP5)
combinatorialcosmicseedfinderP5Top.SeedsFromPositiveY = True
combinatorialcosmicseedfinderP5Top.SeedsFromNegativeY = False
combinatorialcosmicseedfinderP5Top.ClusterCollectionLabel = cms.InputTag("siStripClustersTop")
combinatorialcosmicseedfinderP5Top.OrderedHitsFactoryPSets[0].LayerPSet.TIB1.matchedRecHits = cms.InputTag("siStripMatchedRecHitsTop","matchedRecHit")
combinatorialcosmicseedfinderP5Top.OrderedHitsFactoryPSets[0].LayerPSet.TIB2.matchedRecHits = cms.InputTag("siStripMatchedRecHitsTop","matchedRecHit")
combinatorialcosmicseedfinderP5Top.OrderedHitsFactoryPSets[0].LayerPSet.TIB3.rphiRecHits = cms.InputTag("siStripMatchedRecHitsTop","rphiRecHit")
combinatorialcosmicseedfinderP5Top.OrderedHitsFactoryPSets[0].LayerPSet.TOB1.matchedRecHits = cms.InputTag("siStripMatchedRecHitsTop","matchedRecHit")
combinatorialcosmicseedfinderP5Top.OrderedHitsFactoryPSets[0].LayerPSet.TOB2.matchedRecHits = cms.InputTag("siStripMatchedRecHitsTop","matchedRecHit")
combinatorialcosmicseedfinderP5Top.OrderedHitsFactoryPSets[0].LayerPSet.TOB3.rphiRecHits = cms.InputTag("siStripMatchedRecHitsTop","rphiRecHit")
combinatorialcosmicseedfinderP5Top.OrderedHitsFactoryPSets[0].LayerPSet.TOB4.rphiRecHits = cms.InputTag("siStripMatchedRecHitsTop","rphiRecHit")
combinatorialcosmicseedfinderP5Top.OrderedHitsFactoryPSets[0].LayerPSet.TOB5.rphiRecHits = cms.InputTag("siStripMatchedRecHitsTop","rphiRecHit")
combinatorialcosmicseedfinderP5Top.OrderedHitsFactoryPSets[0].LayerPSet.TOB6.rphiRecHits = cms.InputTag("siStripMatchedRecHitsTop","rphiRecHit")
combinatorialcosmicseedfinderP5Top.OrderedHitsFactoryPSets[0].LayerPSet.TEC.matchedRecHits = cms.InputTag("siStripMatchedRecHitsTop","matchedRecHit")
combinatorialcosmicseedfinderP5Top.OrderedHitsFactoryPSets[0].LayerPSet.TEC.rphiRecHits = cms.InputTag("siStripMatchedRecHitsTop","rphiRecHit")
combinatorialcosmicseedfinderP5Top.OrderedHitsFactoryPSets[1].LayerPSet.TIB1.matchedRecHits = cms.InputTag("siStripMatchedRecHitsTop","matchedRecHit")
combinatorialcosmicseedfinderP5Top.OrderedHitsFactoryPSets[1].LayerPSet.TIB2.matchedRecHits = cms.InputTag("siStripMatchedRecHitsTop","matchedRecHit")
combinatorialcosmicseedfinderP5Top.OrderedHitsFactoryPSets[1].LayerPSet.TIB3.rphiRecHits = cms.InputTag("siStripMatchedRecHitsTop","rphiRecHit")
combinatorialcosmicseedfinderP5Top.OrderedHitsFactoryPSets[1].LayerPSet.TOB1.matchedRecHits = cms.InputTag("siStripMatchedRecHitsTop","matchedRecHit")
combinatorialcosmicseedfinderP5Top.OrderedHitsFactoryPSets[1].LayerPSet.TOB2.matchedRecHits = cms.InputTag("siStripMatchedRecHitsTop","matchedRecHit")
combinatorialcosmicseedfinderP5Top.OrderedHitsFactoryPSets[1].LayerPSet.TOB3.rphiRecHits = cms.InputTag("siStripMatchedRecHitsTop","rphiRecHit")
combinatorialcosmicseedfinderP5Top.OrderedHitsFactoryPSets[1].LayerPSet.TOB4.rphiRecHits = cms.InputTag("siStripMatchedRecHitsTop","rphiRecHit")
combinatorialcosmicseedfinderP5Top.OrderedHitsFactoryPSets[1].LayerPSet.TOB5.rphiRecHits = cms.InputTag("siStripMatchedRecHitsTop","rphiRecHit")
combinatorialcosmicseedfinderP5Top.OrderedHitsFactoryPSets[1].LayerPSet.TOB6.rphiRecHits = cms.InputTag("siStripMatchedRecHitsTop","rphiRecHit")
combinatorialcosmicseedfinderP5Top.OrderedHitsFactoryPSets[1].LayerPSet.TEC.matchedRecHits = cms.InputTag("siStripMatchedRecHitsTop","matchedRecHit")
combinatorialcosmicseedfinderP5Top.OrderedHitsFactoryPSets[1].LayerPSet.TEC.rphiRecHits = cms.InputTag("siStripMatchedRecHitsTop","rphiRecHit")
combinatorialcosmicseedfinderP5Top.OrderedHitsFactoryPSets[2].LayerPSet.TEC.matchedRecHits = cms.InputTag("siStripMatchedRecHitsTop","matchedRecHit")
combinatorialcosmicseedfinderP5Top.OrderedHitsFactoryPSets[2].LayerPSet.TEC.rphiRecHits = cms.InputTag("siStripMatchedRecHitsTop","rphiRecHit")
combinatorialcosmicseedfinderP5Top.OrderedHitsFactoryPSets[3].LayerPSet.TEC.matchedRecHits = cms.InputTag("siStripMatchedRecHitsTop","matchedRecHit")
combinatorialcosmicseedfinderP5Top.OrderedHitsFactoryPSets[3].LayerPSet.TEC.rphiRecHits = cms.InputTag("siStripMatchedRecHitsTop","rphiRecHit")
combinatorialcosmicseedfinderP5Top.OrderedHitsFactoryPSets[4].LayerPSet.TEC.matchedRecHits = cms.InputTag("siStripMatchedRecHitsTop","matchedRecHit")
combinatorialcosmicseedfinderP5Top.OrderedHitsFactoryPSets[4].LayerPSet.TEC.rphiRecHits = cms.InputTag("siStripMatchedRecHitsTop","rphiRecHit")
combinatorialcosmicseedfinderP5Top.OrderedHitsFactoryPSets[5].LayerPSet.TEC.matchedRecHits = cms.InputTag("siStripMatchedRecHitsTop","matchedRecHit")
combinatorialcosmicseedfinderP5Top.OrderedHitsFactoryPSets[5].LayerPSet.TEC.rphiRecHits = cms.InputTag("siStripMatchedRecHitsTop","rphiRecHit")
combinatorialcosmicseedfinderP5Top.MaxNumberOfCosmicClusters = 150
simpleCosmicBONSeedsTop = copy.deepcopy(simpleCosmicBONSeeds)
simpleCosmicBONSeedsTop.PositiveYOnly = True
simpleCosmicBONSeedsTop.NegativeYOnly = False
simpleCosmicBONSeedsTop.ClusterCheckPSet.ClusterCollectionLabel = cms.InputTag("siStripClustersTop")
simpleCosmicBONSeedsTop.TripletsPSet.TIB1.matchedRecHits = cms.InputTag("siStripMatchedRecHitsTop","matchedRecHit")
simpleCosmicBONSeedsTop.TripletsPSet.TIB2.matchedRecHits = cms.InputTag("siStripMatchedRecHitsTop","matchedRecHit")
simpleCosmicBONSeedsTop.TripletsPSet.TIB3.rphiRecHits = cms.InputTag("siStripMatchedRecHitsTop","rphiRecHit")
simpleCosmicBONSeedsTop.TripletsPSet.TOB1.matchedRecHits = cms.InputTag("siStripMatchedRecHitsTop","matchedRecHit")
simpleCosmicBONSeedsTop.TripletsPSet.TOB2.matchedRecHits = cms.InputTag("siStripMatchedRecHitsTop","matchedRecHit")
simpleCosmicBONSeedsTop.TripletsPSet.TOB3.rphiRecHits = cms.InputTag("siStripMatchedRecHitsTop","rphiRecHit")
simpleCosmicBONSeedsTop.TripletsPSet.TOB4.rphiRecHits = cms.InputTag("siStripMatchedRecHitsTop","rphiRecHit")
simpleCosmicBONSeedsTop.TripletsPSet.TOB5.rphiRecHits = cms.InputTag("siStripMatchedRecHitsTop","rphiRecHit")
simpleCosmicBONSeedsTop.TripletsPSet.TOB6.rphiRecHits = cms.InputTag("siStripMatchedRecHitsTop","rphiRecHit")
simpleCosmicBONSeedsTop.TripletsPSet.TEC.matchedRecHits = cms.InputTag("siStripMatchedRecHitsTop","matchedRecHit")
simpleCosmicBONSeedsTop.TripletsPSet.TEC.rphiRecHits = cms.InputTag("siStripMatchedRecHitsTop","rphiRecHit")
simpleCosmicBONSeedsTop.ClusterCheckPSet.MaxNumberOfCosmicClusters = 150
combinedP5SeedsForCTFTop = globalCombinedSeeds.clone(
seedCollections = cms.VInputTag(cms.InputTag('combinatorialcosmicseedfinderP5Top'),cms.InputTag('simpleCosmicBONSeedsTop'))
)
MeasurementTrackerTop = MeasurementTracker.clone(
pixelClusterProducer = cms.string('siPixelClustersTop'),
stripClusterProducer = cms.string('siStripClustersTop'),
ComponentName = cms.string('MeasurementTrackerTop')
)
GroupedCkfTrajectoryBuilderP5Top = copy.deepcopy(GroupedCkfTrajectoryBuilderP5)
GroupedCkfTrajectoryBuilderP5Top.MeasurementTrackerName = cms.string('MeasurementTrackerTop')
GroupedCkfTrajectoryBuilderP5Top.ComponentName = cms.string('GroupedCkfTrajectoryBuilderP5Top')
ckfTrackCandidatesP5Top = copy.deepcopy(ckfTrackCandidatesP5)
ckfTrackCandidatesP5Top.TrajectoryBuilder = 'GroupedCkfTrajectoryBuilderP5Top'
ckfTrackCandidatesP5Top.NavigationSchool   = 'CosmicNavigationSchool'
ckfTrackCandidatesP5Top.src       = 'combinedP5SeedsForCTFTop' #ok for 32X
#ckfTrackCandidatesP5Top.SeedProducer       = 'combinedP5SeedsForCTFTop' #ok for 22X
ckfTrackCandidatesP5Top.useHitsSplitting = True
ctfWithMaterialTracksP5Top = copy.deepcopy(ctfWithMaterialTracksCosmics)
ctfWithMaterialTracksP5Top.src    = 'ckfTrackCandidatesP5Top'
ctfWithMaterialTracksP5Top.Fitter = 'FittingSmootherRKP5'
ctfWithMaterialTracksP5Top.clusterRemovalInfo = "topBottomClusterInfoProducerTop"
ctftracksP5Top = cms.Sequence(combinatorialcosmicseedfinderP5Top*simpleCosmicBONSeedsTop*
                                       combinedP5SeedsForCTFTop*ckfTrackCandidatesP5Top*
                                       ctfWithMaterialTracksP5Top)


###CKF BOTTOM
combinatorialcosmicseedfinderP5Bottom = copy.deepcopy(combinatorialcosmicseedfinderP5)
combinatorialcosmicseedfinderP5Bottom.SeedsFromPositiveY = False
combinatorialcosmicseedfinderP5Bottom.SeedsFromNegativeY = True
combinatorialcosmicseedfinderP5Bottom.OrderedHitsFactoryPSets[0].PropagationDirection = cms.string('oppositeToMomentum')
combinatorialcosmicseedfinderP5Bottom.OrderedHitsFactoryPSets[1].PropagationDirection = cms.string('oppositeToMomentum')
combinatorialcosmicseedfinderP5Bottom.OrderedHitsFactoryPSets[2].PropagationDirection = cms.string('oppositeToMomentum')
combinatorialcosmicseedfinderP5Bottom.OrderedHitsFactoryPSets[3].PropagationDirection = cms.string('oppositeToMomentum')
combinatorialcosmicseedfinderP5Bottom.OrderedHitsFactoryPSets[4].PropagationDirection = cms.string('oppositeToMomentum')
combinatorialcosmicseedfinderP5Bottom.OrderedHitsFactoryPSets[5].PropagationDirection = cms.string('oppositeToMomentum')
combinatorialcosmicseedfinderP5Bottom.ClusterCollectionLabel = cms.InputTag("siStripClustersBottom")
combinatorialcosmicseedfinderP5Bottom.OrderedHitsFactoryPSets[0].LayerPSet.TIB1.matchedRecHits = cms.InputTag("siStripMatchedRecHitsBottom","matchedRecHit")
combinatorialcosmicseedfinderP5Bottom.OrderedHitsFactoryPSets[0].LayerPSet.TIB2.matchedRecHits = cms.InputTag("siStripMatchedRecHitsBottom","matchedRecHit")
combinatorialcosmicseedfinderP5Bottom.OrderedHitsFactoryPSets[0].LayerPSet.TIB3.rphiRecHits = cms.InputTag("siStripMatchedRecHitsBottom","rphiRecHit")
combinatorialcosmicseedfinderP5Bottom.OrderedHitsFactoryPSets[0].LayerPSet.TOB1.matchedRecHits = cms.InputTag("siStripMatchedRecHitsBottom","matchedRecHit")
combinatorialcosmicseedfinderP5Bottom.OrderedHitsFactoryPSets[0].LayerPSet.TOB2.matchedRecHits = cms.InputTag("siStripMatchedRecHitsBottom","matchedRecHit")
combinatorialcosmicseedfinderP5Bottom.OrderedHitsFactoryPSets[0].LayerPSet.TOB3.rphiRecHits = cms.InputTag("siStripMatchedRecHitsBottom","rphiRecHit")
combinatorialcosmicseedfinderP5Bottom.OrderedHitsFactoryPSets[0].LayerPSet.TOB4.rphiRecHits = cms.InputTag("siStripMatchedRecHitsBottom","rphiRecHit")
combinatorialcosmicseedfinderP5Bottom.OrderedHitsFactoryPSets[0].LayerPSet.TOB5.rphiRecHits = cms.InputTag("siStripMatchedRecHitsBottom","rphiRecHit")
combinatorialcosmicseedfinderP5Bottom.OrderedHitsFactoryPSets[0].LayerPSet.TOB6.rphiRecHits = cms.InputTag("siStripMatchedRecHitsBottom","rphiRecHit")
combinatorialcosmicseedfinderP5Bottom.OrderedHitsFactoryPSets[0].LayerPSet.TEC.matchedRecHits = cms.InputTag("siStripMatchedRecHitsBottom","matchedRecHit")
combinatorialcosmicseedfinderP5Bottom.OrderedHitsFactoryPSets[0].LayerPSet.TEC.rphiRecHits = cms.InputTag("siStripMatchedRecHitsBottom","rphiRecHit")
combinatorialcosmicseedfinderP5Bottom.OrderedHitsFactoryPSets[1].LayerPSet.TIB1.matchedRecHits = cms.InputTag("siStripMatchedRecHitsBottom","matchedRecHit")
combinatorialcosmicseedfinderP5Bottom.OrderedHitsFactoryPSets[1].LayerPSet.TIB2.matchedRecHits = cms.InputTag("siStripMatchedRecHitsBottom","matchedRecHit")
combinatorialcosmicseedfinderP5Bottom.OrderedHitsFactoryPSets[1].LayerPSet.TIB3.rphiRecHits = cms.InputTag("siStripMatchedRecHitsBottom","rphiRecHit")
combinatorialcosmicseedfinderP5Bottom.OrderedHitsFactoryPSets[1].LayerPSet.TOB1.matchedRecHits = cms.InputTag("siStripMatchedRecHitsBottom","matchedRecHit")
combinatorialcosmicseedfinderP5Bottom.OrderedHitsFactoryPSets[1].LayerPSet.TOB2.matchedRecHits = cms.InputTag("siStripMatchedRecHitsBottom","matchedRecHit")
combinatorialcosmicseedfinderP5Bottom.OrderedHitsFactoryPSets[1].LayerPSet.TOB3.rphiRecHits = cms.InputTag("siStripMatchedRecHitsBottom","rphiRecHit")
combinatorialcosmicseedfinderP5Bottom.OrderedHitsFactoryPSets[1].LayerPSet.TOB4.rphiRecHits = cms.InputTag("siStripMatchedRecHitsBottom","rphiRecHit")
combinatorialcosmicseedfinderP5Bottom.OrderedHitsFactoryPSets[1].LayerPSet.TOB5.rphiRecHits = cms.InputTag("siStripMatchedRecHitsBottom","rphiRecHit")
combinatorialcosmicseedfinderP5Bottom.OrderedHitsFactoryPSets[1].LayerPSet.TOB6.rphiRecHits = cms.InputTag("siStripMatchedRecHitsBottom","rphiRecHit")
combinatorialcosmicseedfinderP5Bottom.OrderedHitsFactoryPSets[1].LayerPSet.TEC.matchedRecHits = cms.InputTag("siStripMatchedRecHitsBottom","matchedRecHit")
combinatorialcosmicseedfinderP5Bottom.OrderedHitsFactoryPSets[1].LayerPSet.TEC.rphiRecHits = cms.InputTag("siStripMatchedRecHitsBottom","rphiRecHit")
combinatorialcosmicseedfinderP5Bottom.OrderedHitsFactoryPSets[2].LayerPSet.TEC.matchedRecHits = cms.InputTag("siStripMatchedRecHitsBottom","matchedRecHit")
combinatorialcosmicseedfinderP5Bottom.OrderedHitsFactoryPSets[2].LayerPSet.TEC.rphiRecHits = cms.InputTag("siStripMatchedRecHitsBottom","rphiRecHit")
combinatorialcosmicseedfinderP5Bottom.OrderedHitsFactoryPSets[3].LayerPSet.TEC.matchedRecHits = cms.InputTag("siStripMatchedRecHitsBottom","matchedRecHit")
combinatorialcosmicseedfinderP5Bottom.OrderedHitsFactoryPSets[3].LayerPSet.TEC.rphiRecHits = cms.InputTag("siStripMatchedRecHitsBottom","rphiRecHit")
combinatorialcosmicseedfinderP5Bottom.OrderedHitsFactoryPSets[4].LayerPSet.TEC.matchedRecHits = cms.InputTag("siStripMatchedRecHitsBottom","matchedRecHit")
combinatorialcosmicseedfinderP5Bottom.OrderedHitsFactoryPSets[4].LayerPSet.TEC.rphiRecHits = cms.InputTag("siStripMatchedRecHitsBottom","rphiRecHit")
combinatorialcosmicseedfinderP5Bottom.OrderedHitsFactoryPSets[5].LayerPSet.TEC.matchedRecHits = cms.InputTag("siStripMatchedRecHitsBottom","matchedRecHit")
combinatorialcosmicseedfinderP5Bottom.OrderedHitsFactoryPSets[5].LayerPSet.TEC.rphiRecHits = cms.InputTag("siStripMatchedRecHitsBottom","rphiRecHit")
combinatorialcosmicseedfinderP5Bottom.MaxNumberOfCosmicClusters = 150
simpleCosmicBONSeedsBottom = copy.deepcopy(simpleCosmicBONSeeds)
simpleCosmicBONSeedsBottom.PositiveYOnly = False
simpleCosmicBONSeedsBottom.NegativeYOnly = True
simpleCosmicBONSeedsBottom.ClusterCheckPSet.ClusterCollectionLabel = cms.InputTag("siStripClustersBottom")
simpleCosmicBONSeedsBottom.TripletsPSet.TIB1.matchedRecHits = cms.InputTag("siStripMatchedRecHitsBottom","matchedRecHit")
simpleCosmicBONSeedsBottom.TripletsPSet.TIB2.matchedRecHits = cms.InputTag("siStripMatchedRecHitsBottom","matchedRecHit")
simpleCosmicBONSeedsBottom.TripletsPSet.TIB3.rphiRecHits = cms.InputTag("siStripMatchedRecHitsBottom","rphiRecHit")
simpleCosmicBONSeedsBottom.TripletsPSet.TOB1.matchedRecHits = cms.InputTag("siStripMatchedRecHitsBottom","matchedRecHit")
simpleCosmicBONSeedsBottom.TripletsPSet.TOB2.matchedRecHits = cms.InputTag("siStripMatchedRecHitsBottom","matchedRecHit")
simpleCosmicBONSeedsBottom.TripletsPSet.TOB3.rphiRecHits = cms.InputTag("siStripMatchedRecHitsBottom","rphiRecHit")
simpleCosmicBONSeedsBottom.TripletsPSet.TOB4.rphiRecHits = cms.InputTag("siStripMatchedRecHitsBottom","rphiRecHit")
simpleCosmicBONSeedsBottom.TripletsPSet.TOB5.rphiRecHits = cms.InputTag("siStripMatchedRecHitsBottom","rphiRecHit")
simpleCosmicBONSeedsBottom.TripletsPSet.TOB6.rphiRecHits = cms.InputTag("siStripMatchedRecHitsBottom","rphiRecHit")
simpleCosmicBONSeedsBottom.TripletsPSet.TEC.matchedRecHits = cms.InputTag("siStripMatchedRecHitsBottom","matchedRecHit")
simpleCosmicBONSeedsBottom.TripletsPSet.TEC.rphiRecHits = cms.InputTag("siStripMatchedRecHitsBottom","rphiRecHit")
simpleCosmicBONSeedsBottom.ClusterCheckPSet.MaxNumberOfCosmicClusters = 150
combinedP5SeedsForCTFBottom = globalCombinedSeeds.clone(
seedCollections = cms.VInputTag(cms.InputTag('combinatorialcosmicseedfinderP5Bottom'),cms.InputTag('simpleCosmicBONSeedsBottom'))
)
MeasurementTrackerBottom = MeasurementTracker.clone(
pixelClusterProducer = cms.string('siPixelClustersBottom'),
stripClusterProducer = cms.string('siStripClustersBottom'),
ComponentName = cms.string('MeasurementTrackerBottom')
)
GroupedCkfTrajectoryBuilderP5Bottom = copy.deepcopy(GroupedCkfTrajectoryBuilderP5)
GroupedCkfTrajectoryBuilderP5Bottom.MeasurementTrackerName = cms.string('MeasurementTrackerBottom')
GroupedCkfTrajectoryBuilderP5Bottom.ComponentName = cms.string('GroupedCkfTrajectoryBuilderP5Bottom')
ckfTrackCandidatesP5Bottom = copy.deepcopy(ckfTrackCandidatesP5)
ckfTrackCandidatesP5Bottom.TrajectoryBuilder = 'GroupedCkfTrajectoryBuilderP5Bottom'
ckfTrackCandidatesP5Bottom.NavigationSchool   = 'CosmicNavigationSchool'
ckfTrackCandidatesP5Bottom.src       = 'combinedP5SeedsForCTFBottom' #ok for 32X
#ckfTrackCandidatesP5Bottom.SeedProducer       = 'combinedP5SeedsForCTFBottom' #ok for 22X
ckfTrackCandidatesP5Bottom.useHitsSplitting = True
ctfWithMaterialTracksP5Bottom = copy.deepcopy(ctfWithMaterialTracksCosmics)
ctfWithMaterialTracksP5Bottom.src    = 'ckfTrackCandidatesP5Bottom'
ctfWithMaterialTracksP5Bottom.Fitter = 'FittingSmootherRKP5'
ctfWithMaterialTracksP5Bottom.clusterRemovalInfo = "topBottomClusterInfoProducerBottom"
ctftracksP5Bottom = cms.Sequence(combinatorialcosmicseedfinderP5Bottom*simpleCosmicBONSeedsBottom*
                                       combinedP5SeedsForCTFBottom*ckfTrackCandidatesP5Bottom*
                                       ctfWithMaterialTracksP5Bottom)

#COSMIC TOP
cosmicseedfinderP5Top       = copy.deepcopy(cosmicseedfinderP5)
cosmicCandidateFinderP5Top  = copy.deepcopy(cosmicCandidateFinderP5)
cosmictrackfinderP5Top      = copy.deepcopy(cosmictrackfinderCosmics)
cosmicseedfinderP5Top.stereorecHits = cms.InputTag("siStripMatchedRecHitsTop","stereoRecHit")
cosmicseedfinderP5Top.matchedRecHits = cms.InputTag("siStripMatchedRecHitsTop","matchedRecHit")
cosmicseedfinderP5Top.rphirecHits = cms.InputTag("siStripMatchedRecHitsTop","rphiRecHit")
cosmicseedfinderP5Top.ClusterCollectionLabel = cms.InputTag("siStripClustersTop")
cosmicseedfinderP5Top.PositiveYOnly = True
cosmicseedfinderP5Top.NegativeYOnly = False
cosmicseedfinderP5Top.MaxNumberOfCosmicClusters = 150
cosmicCandidateFinderP5Top.cosmicSeeds = 'cosmicseedfinderP5Top'
cosmicCandidateFinderP5Top.stereorecHits = cms.InputTag("siStripMatchedRecHitsTop","stereoRecHit")
cosmicCandidateFinderP5Top.HitProducer = cms.string('siStripRecHitsTop')
cosmicCandidateFinderP5Top.pixelRecHits = cms.InputTag("siPixelRecHitsTop")
cosmicCandidateFinderP5Top.matchedRecHits = cms.InputTag("siStripMatchedRecHitsTop","matchedRecHit")
cosmicCandidateFinderP5Top.rphirecHits = cms.InputTag("siStripMatchedRecHitsTop","rphiRecHit")
cosmictrackfinderP5Top.src = 'cosmicCandidateFinderP5Top'
cosmictrackfinderP5Top.clusterRemovalInfo = "topBottomClusterInfoProducerTop"
cosmictracksP5Top = cms.Sequence(cosmicseedfinderP5Top*cosmicCandidateFinderP5Top*cosmictrackfinderP5Top)

#COSMIC BOTTOM
cosmicseedfinderP5Bottom       = copy.deepcopy(cosmicseedfinderP5)
cosmicCandidateFinderP5Bottom  = copy.deepcopy(cosmicCandidateFinderP5)
cosmictrackfinderP5Bottom      = copy.deepcopy(cosmictrackfinderCosmics)
cosmicseedfinderP5Bottom.stereorecHits = cms.InputTag("siStripMatchedRecHitsBottom","stereoRecHit")
cosmicseedfinderP5Bottom.matchedRecHits = cms.InputTag("siStripMatchedRecHitsBottom","matchedRecHit")
cosmicseedfinderP5Bottom.rphirecHits = cms.InputTag("siStripMatchedRecHitsBottom","rphiRecHit")
cosmicseedfinderP5Bottom.ClusterCollectionLabel = cms.InputTag("siStripClustersBottom")
cosmicseedfinderP5Bottom.PositiveYOnly = False
cosmicseedfinderP5Bottom.NegativeYOnly = True
cosmicseedfinderP5Bottom.MaxNumberOfCosmicClusters = 150
cosmicCandidateFinderP5Bottom.cosmicSeeds = 'cosmicseedfinderP5Bottom'
cosmicCandidateFinderP5Bottom.stereorecHits = cms.InputTag("siStripMatchedRecHitsBottom","stereoRecHit")
cosmicCandidateFinderP5Bottom.HitProducer = cms.string('siStripRecHitsBottom')
cosmicCandidateFinderP5Bottom.pixelRecHits = cms.InputTag("siPixelRecHitsBottom")
cosmicCandidateFinderP5Bottom.matchedRecHits = cms.InputTag("siStripMatchedRecHitsBottom","matchedRecHit")
cosmicCandidateFinderP5Bottom.rphirecHits = cms.InputTag("siStripMatchedRecHitsBottom","rphiRecHit")
cosmictrackfinderP5Bottom.src = 'cosmicCandidateFinderP5Bottom'
cosmictrackfinderP5Bottom.clusterRemovalInfo = "topBottomClusterInfoProducerBottom"
cosmictracksP5Bottom = cms.Sequence(cosmicseedfinderP5Bottom*cosmicCandidateFinderP5Bottom*cosmictrackfinderP5Bottom)

#RS TOP
roadSearchSeedsP5Top      = copy.deepcopy(roadSearchSeedsP5)
roadSearchCloudsP5Top     = copy.deepcopy(roadSearchCloudsP5)
rsTrackCandidatesP5Top    = copy.deepcopy(rsTrackCandidatesP5)
rsWithMaterialTracksP5Top = copy.deepcopy(rsWithMaterialTracksCosmics)
roadSearchSeedsP5Top.AllPositiveOnly = True
roadSearchSeedsP5Top.pixelRecHits = cms.InputTag("siPixelRecHitsTop")
roadSearchSeedsP5Top.rphiStripRecHits = cms.InputTag("siStripMatchedRecHitsTop","rphiRecHit")
roadSearchSeedsP5Top.stereoStripRecHits = cms.InputTag("siStripMatchedRecHitsTop","stereoRecHit")
roadSearchSeedsP5Top.ClusterCollectionLabel = cms.InputTag("siStripClustersTop")
roadSearchSeedsP5Top.matchedStripRecHits = cms.InputTag("siStripMatchedRecHitsTop","matchedRecHit")
roadSearchSeedsP5Top.MaxNumberOfCosmicClusters = 150
roadSearchCloudsP5Top.SeedProducer = 'roadSearchSeedsP5Top'
roadSearchCloudsP5Top.pixelRecHits = cms.InputTag("siPixelRecHitsTop")
roadSearchCloudsP5Top.rphiStripRecHits = cms.InputTag("siStripMatchedRecHitsTop","rphiRecHit")
roadSearchCloudsP5Top.stereoStripRecHits = cms.InputTag("siStripMatchedRecHitsTop","stereoRecHit")
roadSearchCloudsP5Top.ClusterCollectionLabel = cms.InputTag("siStripClustersTop")
roadSearchCloudsP5Top.matchedStripRecHits = cms.InputTag("siStripMatchedRecHitsTop","matchedRecHit")
rsTrackCandidatesP5Top.CloudProducer = 'roadSearchCloudsP5Top'
rsTrackCandidatesP5Top.SplitMatchedHits = True
rsWithMaterialTracksP5Top.src = 'rsTrackCandidatesP5Top'
rsWithMaterialTracksP5Top.clusterRemovalInfo = "topBottomClusterInfoProducerTop"
rstracksP5Top = cms.Sequence(roadSearchSeedsP5Top*roadSearchCloudsP5Top*
                                     rsTrackCandidatesP5Top*rsWithMaterialTracksP5Top)

#RS BOTTOM
roadSearchSeedsP5Bottom      = copy.deepcopy(roadSearchSeedsP5)
roadSearchCloudsP5Bottom     = copy.deepcopy(roadSearchCloudsP5)
rsTrackCandidatesP5Bottom    = copy.deepcopy(rsTrackCandidatesP5)
rsWithMaterialTracksP5Bottom = copy.deepcopy(rsWithMaterialTracksCosmics)
roadSearchSeedsP5Bottom.AllNegativeOnly = True
roadSearchSeedsP5Bottom.pixelRecHits = cms.InputTag("siPixelRecHitsBottom")
roadSearchSeedsP5Bottom.rphiStripRecHits = cms.InputTag("siStripMatchedRecHitsBottom","rphiRecHit")
roadSearchSeedsP5Bottom.stereoStripRecHits = cms.InputTag("siStripMatchedRecHitsBottom","stereoRecHit")
roadSearchSeedsP5Bottom.ClusterCollectionLabel = cms.InputTag("siStripClustersBottom")
roadSearchSeedsP5Bottom.matchedStripRecHits = cms.InputTag("siStripMatchedRecHitsBottom","matchedRecHit")
roadSearchSeedsP5Bottom.MaxNumberOfCosmicClusters = 150
roadSearchCloudsP5Bottom.SeedProducer = 'roadSearchSeedsP5Bottom'
roadSearchCloudsP5Bottom.pixelRecHits = cms.InputTag("siPixelRecHitsBottom")
roadSearchCloudsP5Bottom.rphiStripRecHits = cms.InputTag("siStripMatchedRecHitsBottom","rphiRecHit")
roadSearchCloudsP5Bottom.stereoStripRecHits = cms.InputTag("siStripMatchedRecHitsBottom","stereoRecHit")
roadSearchCloudsP5Bottom.ClusterCollectionLabel = cms.InputTag("siStripClustersBottom")
roadSearchCloudsP5Bottom.matchedStripRecHits = cms.InputTag("siStripMatchedRecHitsBottom","matchedRecHit")
rsTrackCandidatesP5Bottom.CloudProducer = 'roadSearchCloudsP5Bottom'
rsTrackCandidatesP5Bottom.SplitMatchedHits = True
rsWithMaterialTracksP5Bottom.src = 'rsTrackCandidatesP5Bottom'
rsWithMaterialTracksP5Bottom.clusterRemovalInfo = "topBottomClusterInfoProducerBottom"
rstracksP5Bottom = cms.Sequence(roadSearchSeedsP5Bottom*roadSearchCloudsP5Bottom*
                                         rsTrackCandidatesP5Bottom*rsWithMaterialTracksP5Bottom)

#TOP SEQUENCE
# (SK) keep rstracks commented out in case of resurrection
#tracksP5Top = cms.Sequence(ctftracksP5Top+cosmictracksP5Top+rstracksP5Top)
tracksP5Top = cms.Sequence(ctftracksP5Top+cosmictracksP5Top)
#BOTTOM SEQUENCE
# (SK) keep rstracks commented out in case of resurrection
#tracksP5Bottom = cms.Sequence(ctftracksP5Bottom+cosmictracksP5Bottom+rstracksP5Bottom)
tracksP5Bottom = cms.Sequence(ctftracksP5Bottom+cosmictracksP5Bottom)
