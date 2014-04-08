import FWCore.ParameterSet.Config as cms
import copy

from RecoLocalTracker.SubCollectionProducers.ClusterSelectorTopBottom_cfi import *
from RecoLocalTracker.Configuration.RecoLocalTracker_Cosmics_cff import *
from RecoTracker.SpecialSeedGenerators.CombinatorialSeedGeneratorForCosmicsP5_cff import combinatorialcosmicseedfinderP5, combinatorialcosmicseedingtripletsP5, combinatorialcosmicseedingpairsTOBP5, combinatorialcosmicseedingpairsTECposP5, combinatorialcosmicseedingpairsTECnegP5
from RecoTracker.SpecialSeedGenerators.SimpleCosmicBONSeeder_cff import simpleCosmicBONSeeds, simpleCosmicBONSeedingLayers
from RecoTracker.TkSeedGenerator.GlobalCombinedSeeds_cff import globalCombinedSeeds
from RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cff import MeasurementTracker
from RecoTracker.CkfPattern.CkfTrackCandidatesP5_cff import GroupedCkfTrajectoryBuilderP5
from RecoTracker.CkfPattern.CkfTrackCandidatesP5_cff import ckfTrackCandidatesP5
from RecoTracker.TrackProducer.CTFFinalFitWithMaterialP5_cff import ctfWithMaterialTracksCosmics
from RecoTracker.SpecialSeedGenerators.CosmicSeedP5Pairs_cff import cosmicseedfinderP5
from RecoTracker.SingleTrackPattern.CosmicTrackFinderP5_cff import cosmicCandidateFinderP5
from RecoTracker.SingleTrackPattern.CosmicTrackFinderP5_cff import cosmictrackfinderCosmics


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
combinatorialcosmicseedingtripletsP5Top = copy.deepcopy(combinatorialcosmicseedingtripletsP5)
combinatorialcosmicseedingtripletsP5Top.TIB1.matchedRecHits = cms.InputTag("siStripMatchedRecHitsTop","matchedRecHit")
combinatorialcosmicseedingtripletsP5Top.TIB2.matchedRecHits = cms.InputTag("siStripMatchedRecHitsTop","matchedRecHit")
combinatorialcosmicseedingtripletsP5Top.TIB3.rphiRecHits = cms.InputTag("siStripMatchedRecHitsTop","rphiRecHit")
combinatorialcosmicseedingtripletsP5Top.TOB1.matchedRecHits = cms.InputTag("siStripMatchedRecHitsTop","matchedRecHit")
combinatorialcosmicseedingtripletsP5Top.TOB2.matchedRecHits = cms.InputTag("siStripMatchedRecHitsTop","matchedRecHit")
combinatorialcosmicseedingtripletsP5Top.TOB3.rphiRecHits = cms.InputTag("siStripMatchedRecHitsTop","rphiRecHit")
combinatorialcosmicseedingtripletsP5Top.TOB4.rphiRecHits = cms.InputTag("siStripMatchedRecHitsTop","rphiRecHit")
combinatorialcosmicseedingtripletsP5Top.TOB5.rphiRecHits = cms.InputTag("siStripMatchedRecHitsTop","rphiRecHit")
combinatorialcosmicseedingtripletsP5Top.TOB6.rphiRecHits = cms.InputTag("siStripMatchedRecHitsTop","rphiRecHit")
combinatorialcosmicseedingtripletsP5Top.TEC.matchedRecHits = cms.InputTag("siStripMatchedRecHitsTop","matchedRecHit")
combinatorialcosmicseedingtripletsP5Top.TEC.rphiRecHits = cms.InputTag("siStripMatchedRecHitsTop","rphiRecHit")
combinatorialcosmicseedingpairsTOBP5Top = copy.deepcopy(combinatorialcosmicseedingpairsTOBP5)
combinatorialcosmicseedingpairsTOBP5Top.TIB1.matchedRecHits = cms.InputTag("siStripMatchedRecHitsTop","matchedRecHit")
combinatorialcosmicseedingpairsTOBP5Top.TIB2.matchedRecHits = cms.InputTag("siStripMatchedRecHitsTop","matchedRecHit")
combinatorialcosmicseedingpairsTOBP5Top.TIB3.rphiRecHits = cms.InputTag("siStripMatchedRecHitsTop","rphiRecHit")
combinatorialcosmicseedingpairsTOBP5Top.TOB1.matchedRecHits = cms.InputTag("siStripMatchedRecHitsTop","matchedRecHit")
combinatorialcosmicseedingpairsTOBP5Top.TOB2.matchedRecHits = cms.InputTag("siStripMatchedRecHitsTop","matchedRecHit")
combinatorialcosmicseedingpairsTOBP5Top.TOB3.rphiRecHits = cms.InputTag("siStripMatchedRecHitsTop","rphiRecHit")
combinatorialcosmicseedingpairsTOBP5Top.TOB4.rphiRecHits = cms.InputTag("siStripMatchedRecHitsTop","rphiRecHit")
combinatorialcosmicseedingpairsTOBP5Top.TOB5.rphiRecHits = cms.InputTag("siStripMatchedRecHitsTop","rphiRecHit")
combinatorialcosmicseedingpairsTOBP5Top.TOB6.rphiRecHits = cms.InputTag("siStripMatchedRecHitsTop","rphiRecHit")
combinatorialcosmicseedingpairsTOBP5Top.TEC.matchedRecHits = cms.InputTag("siStripMatchedRecHitsTop","matchedRecHit")
combinatorialcosmicseedingpairsTOBP5Top.TEC.rphiRecHits = cms.InputTag("siStripMatchedRecHitsTop","rphiRecHit")
combinatorialcosmicseedingpairsTECposP5Top = copy.deepcopy(combinatorialcosmicseedingpairsTECposP5)
combinatorialcosmicseedingpairsTECposP5Top.TEC.matchedRecHits = cms.InputTag("siStripMatchedRecHitsTop","matchedRecHit")
combinatorialcosmicseedingpairsTECposP5Top.TEC.rphiRecHits = cms.InputTag("siStripMatchedRecHitsTop","rphiRecHit")
combinatorialcosmicseedingpairsTECnegP5Top = copy.deepcopy(combinatorialcosmicseedingpairsTECnegP5)
combinatorialcosmicseedingpairsTECnegP5Top.TEC.matchedRecHits = cms.InputTag("siStripMatchedRecHitsTop","matchedRecHit")
combinatorialcosmicseedingpairsTECnegP5Top.TEC.rphiRecHits = cms.InputTag("siStripMatchedRecHitsTop","rphiRecHit")
combinatorialcosmicseedfinderP5Top = copy.deepcopy(combinatorialcosmicseedfinderP5)
combinatorialcosmicseedfinderP5Top.SeedsFromPositiveY = True
combinatorialcosmicseedfinderP5Top.SeedsFromNegativeY = False
combinatorialcosmicseedfinderP5Top.ClusterCollectionLabel = cms.InputTag("siStripClustersTop")
combinatorialcosmicseedfinderP5Top.OrderedHitsFactoryPSets[0].LayerSrc = "combinatorialcosmicseedingtripletsP5Top"
combinatorialcosmicseedfinderP5Top.OrderedHitsFactoryPSets[1].LayerSrc = "combinatorialcosmicseedingpairsTOBP5Top"
combinatorialcosmicseedfinderP5Top.OrderedHitsFactoryPSets[2].LayerSrc = "combinatorialcosmicseedingpairsTECposP5Top"
combinatorialcosmicseedfinderP5Top.OrderedHitsFactoryPSets[3].LayerSrc = "combinatorialcosmicseedingpairsTECposP5Top"
combinatorialcosmicseedfinderP5Top.OrderedHitsFactoryPSets[4].LayerSrc = "combinatorialcosmicseedingpairsTECnegP5Top"
combinatorialcosmicseedfinderP5Top.OrderedHitsFactoryPSets[5].LayerSrc = "combinatorialcosmicseedingpairsTECnegP5Top"
combinatorialcosmicseedfinderP5Top.MaxNumberOfCosmicClusters = 150
simpleCosmicBONSeedingLayersTop = copy.deepcopy(simpleCosmicBONSeedingLayers)
simpleCosmicBONSeedingLayersTop.TIB1.matchedRecHits = cms.InputTag("siStripMatchedRecHitsTop","matchedRecHit")
simpleCosmicBONSeedingLayersTop.TIB2.matchedRecHits = cms.InputTag("siStripMatchedRecHitsTop","matchedRecHit")
simpleCosmicBONSeedingLayersTop.TIB3.rphiRecHits = cms.InputTag("siStripMatchedRecHitsTop","rphiRecHit")
simpleCosmicBONSeedingLayersTop.TOB1.matchedRecHits = cms.InputTag("siStripMatchedRecHitsTop","matchedRecHit")
simpleCosmicBONSeedingLayersTop.TOB2.matchedRecHits = cms.InputTag("siStripMatchedRecHitsTop","matchedRecHit")
simpleCosmicBONSeedingLayersTop.TOB3.rphiRecHits = cms.InputTag("siStripMatchedRecHitsTop","rphiRecHit")
simpleCosmicBONSeedingLayersTop.TOB4.rphiRecHits = cms.InputTag("siStripMatchedRecHitsTop","rphiRecHit")
simpleCosmicBONSeedingLayersTop.TOB5.rphiRecHits = cms.InputTag("siStripMatchedRecHitsTop","rphiRecHit")
simpleCosmicBONSeedingLayersTop.TOB6.rphiRecHits = cms.InputTag("siStripMatchedRecHitsTop","rphiRecHit")
simpleCosmicBONSeedingLayersTop.TEC.matchedRecHits = cms.InputTag("siStripMatchedRecHitsTop","matchedRecHit")
simpleCosmicBONSeedingLayersTop.TEC.rphiRecHits = cms.InputTag("siStripMatchedRecHitsTop","rphiRecHit")
simpleCosmicBONSeedsTop = copy.deepcopy(simpleCosmicBONSeeds)
simpleCosmicBONSeedsTop.PositiveYOnly = True
simpleCosmicBONSeedsTop.NegativeYOnly = False
simpleCosmicBONSeedsTop.ClusterCheckPSet.ClusterCollectionLabel = cms.InputTag("siStripClustersTop")
simpleCosmicBONSeedsTop.TripletsSrc = "simpleCosmicBONSeedingLayersTop"
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
ckfTrackCandidatesP5Top = copy.deepcopy(ckfTrackCandidatesP5)
ckfTrackCandidatesP5Top.TrajectoryBuilderPSet.refToPSet_ = 'GroupedCkfTrajectoryBuilderP5Top'
ckfTrackCandidatesP5Top.NavigationSchool   = 'CosmicNavigationSchool'
ckfTrackCandidatesP5Top.src       = 'combinedP5SeedsForCTFTop' #ok for 32X
#ckfTrackCandidatesP5Top.SeedProducer       = 'combinedP5SeedsForCTFTop' #ok for 22X
ckfTrackCandidatesP5Top.useHitsSplitting = True
ctfWithMaterialTracksP5Top = copy.deepcopy(ctfWithMaterialTracksCosmics)
ctfWithMaterialTracksP5Top.src    = 'ckfTrackCandidatesP5Top'
ctfWithMaterialTracksP5Top.Fitter = 'FittingSmootherRKP5'
ctfWithMaterialTracksP5Top.clusterRemovalInfo = "topBottomClusterInfoProducerTop"
ctftracksP5Top = cms.Sequence(combinatorialcosmicseedingtripletsP5Top*combinatorialcosmicseedingpairsTOBP5Top*
                              combinatorialcosmicseedingpairsTECposP5Top*combinatorialcosmicseedingpairsTECnegP5Top*
                              combinatorialcosmicseedfinderP5Top*simpleCosmicBONSeedingLayersTop*simpleCosmicBONSeedsTop*
                                       combinedP5SeedsForCTFTop*ckfTrackCandidatesP5Top*
                                       ctfWithMaterialTracksP5Top)


###CKF BOTTOM
combinatorialcosmicseedingtripletsP5Bottom = copy.deepcopy(combinatorialcosmicseedingtripletsP5)
combinatorialcosmicseedingtripletsP5Bottom.TIB1.matchedRecHits = cms.InputTag("siStripMatchedRecHitsBottom","matchedRecHit")
combinatorialcosmicseedingtripletsP5Bottom.TIB2.matchedRecHits = cms.InputTag("siStripMatchedRecHitsBottom","matchedRecHit")
combinatorialcosmicseedingtripletsP5Bottom.TIB3.rphiRecHits = cms.InputTag("siStripMatchedRecHitsBottom","rphiRecHit")
combinatorialcosmicseedingtripletsP5Bottom.TOB1.matchedRecHits = cms.InputTag("siStripMatchedRecHitsBottom","matchedRecHit")
combinatorialcosmicseedingtripletsP5Bottom.TOB2.matchedRecHits = cms.InputTag("siStripMatchedRecHitsBottom","matchedRecHit")
combinatorialcosmicseedingtripletsP5Bottom.TOB3.rphiRecHits = cms.InputTag("siStripMatchedRecHitsBottom","rphiRecHit")
combinatorialcosmicseedingtripletsP5Bottom.TOB4.rphiRecHits = cms.InputTag("siStripMatchedRecHitsBottom","rphiRecHit")
combinatorialcosmicseedingtripletsP5Bottom.TOB5.rphiRecHits = cms.InputTag("siStripMatchedRecHitsBottom","rphiRecHit")
combinatorialcosmicseedingtripletsP5Bottom.TOB6.rphiRecHits = cms.InputTag("siStripMatchedRecHitsBottom","rphiRecHit")
combinatorialcosmicseedingtripletsP5Bottom.TEC.matchedRecHits = cms.InputTag("siStripMatchedRecHitsBottom","matchedRecHit")
combinatorialcosmicseedingtripletsP5Bottom.TEC.rphiRecHits = cms.InputTag("siStripMatchedRecHitsBottom","rphiRecHit")
combinatorialcosmicseedingpairsTOBP5Bottom = copy.deepcopy(combinatorialcosmicseedingpairsTOBP5)
combinatorialcosmicseedingpairsTOBP5Bottom.TIB1.matchedRecHits = cms.InputTag("siStripMatchedRecHitsBottom","matchedRecHit")
combinatorialcosmicseedingpairsTOBP5Bottom.TIB2.matchedRecHits = cms.InputTag("siStripMatchedRecHitsBottom","matchedRecHit")
combinatorialcosmicseedingpairsTOBP5Bottom.TIB3.rphiRecHits = cms.InputTag("siStripMatchedRecHitsBottom","rphiRecHit")
combinatorialcosmicseedingpairsTOBP5Bottom.TOB1.matchedRecHits = cms.InputTag("siStripMatchedRecHitsBottom","matchedRecHit")
combinatorialcosmicseedingpairsTOBP5Bottom.TOB2.matchedRecHits = cms.InputTag("siStripMatchedRecHitsBottom","matchedRecHit")
combinatorialcosmicseedingpairsTOBP5Bottom.TOB3.rphiRecHits = cms.InputTag("siStripMatchedRecHitsBottom","rphiRecHit")
combinatorialcosmicseedingpairsTOBP5Bottom.TOB4.rphiRecHits = cms.InputTag("siStripMatchedRecHitsBottom","rphiRecHit")
combinatorialcosmicseedingpairsTOBP5Bottom.TOB5.rphiRecHits = cms.InputTag("siStripMatchedRecHitsBottom","rphiRecHit")
combinatorialcosmicseedingpairsTOBP5Bottom.TOB6.rphiRecHits = cms.InputTag("siStripMatchedRecHitsBottom","rphiRecHit")
combinatorialcosmicseedingpairsTOBP5Bottom.TEC.matchedRecHits = cms.InputTag("siStripMatchedRecHitsBottom","matchedRecHit")
combinatorialcosmicseedingpairsTOBP5Bottom.TEC.rphiRecHits = cms.InputTag("siStripMatchedRecHitsBottom","rphiRecHit")
combinatorialcosmicseedingpairsTECposP5Bottom = copy.deepcopy(combinatorialcosmicseedingpairsTECposP5)
combinatorialcosmicseedingpairsTECposP5Bottom.TEC.matchedRecHits = cms.InputTag("siStripMatchedRecHitsBottom","matchedRecHit")
combinatorialcosmicseedingpairsTECposP5Bottom.TEC.rphiRecHits = cms.InputTag("siStripMatchedRecHitsBottom","rphiRecHit")
combinatorialcosmicseedingpairsTECnegP5Bottom = copy.deepcopy(combinatorialcosmicseedingpairsTECnegP5)
combinatorialcosmicseedingpairsTECnegP5Bottom.TEC.matchedRecHits = cms.InputTag("siStripMatchedRecHitsBottom","matchedRecHit")
combinatorialcosmicseedingpairsTECnegP5Bottom.TEC.rphiRecHits = cms.InputTag("siStripMatchedRecHitsBottom","rphiRecHit")
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
combinatorialcosmicseedfinderP5Bottom.OrderedHitsFactoryPSets[0].LayerSrc = "combinatorialcosmicseedingtripletsP5Bottom"
combinatorialcosmicseedfinderP5Bottom.OrderedHitsFactoryPSets[1].LayerSrc = "combinatorialcosmicseedingpairsTOBP5Bottom"
combinatorialcosmicseedfinderP5Bottom.OrderedHitsFactoryPSets[2].LayerSrc = "combinatorialcosmicseedingpairsTECposP5Bottom"
combinatorialcosmicseedfinderP5Bottom.OrderedHitsFactoryPSets[3].LayerSrc = "combinatorialcosmicseedingpairsTECposP5Bottom"
combinatorialcosmicseedfinderP5Bottom.OrderedHitsFactoryPSets[4].LayerSrc = "combinatorialcosmicseedingpairsTECnegP5Bottom"
combinatorialcosmicseedfinderP5Bottom.OrderedHitsFactoryPSets[5].LayerSrc = "combinatorialcosmicseedingpairsTECnegP5Bottom"
combinatorialcosmicseedfinderP5Bottom.MaxNumberOfCosmicClusters = 150
simpleCosmicBONSeedingLayersBottom = copy.deepcopy(simpleCosmicBONSeedingLayers)
simpleCosmicBONSeedingLayersBottom.TIB1.matchedRecHits = cms.InputTag("siStripMatchedRecHitsBottom","matchedRecHit")
simpleCosmicBONSeedingLayersBottom.TIB2.matchedRecHits = cms.InputTag("siStripMatchedRecHitsBottom","matchedRecHit")
simpleCosmicBONSeedingLayersBottom.TIB3.rphiRecHits = cms.InputTag("siStripMatchedRecHitsBottom","rphiRecHit")
simpleCosmicBONSeedingLayersBottom.TOB1.matchedRecHits = cms.InputTag("siStripMatchedRecHitsBottom","matchedRecHit")
simpleCosmicBONSeedingLayersBottom.TOB2.matchedRecHits = cms.InputTag("siStripMatchedRecHitsBottom","matchedRecHit")
simpleCosmicBONSeedingLayersBottom.TOB3.rphiRecHits = cms.InputTag("siStripMatchedRecHitsBottom","rphiRecHit")
simpleCosmicBONSeedingLayersBottom.TOB4.rphiRecHits = cms.InputTag("siStripMatchedRecHitsBottom","rphiRecHit")
simpleCosmicBONSeedingLayersBottom.TOB5.rphiRecHits = cms.InputTag("siStripMatchedRecHitsBottom","rphiRecHit")
simpleCosmicBONSeedingLayersBottom.TOB6.rphiRecHits = cms.InputTag("siStripMatchedRecHitsBottom","rphiRecHit")
simpleCosmicBONSeedingLayersBottom.TEC.matchedRecHits = cms.InputTag("siStripMatchedRecHitsBottom","matchedRecHit")
simpleCosmicBONSeedingLayersBottom.TEC.rphiRecHits = cms.InputTag("siStripMatchedRecHitsBottom","rphiRecHit")
simpleCosmicBONSeedsBottom = copy.deepcopy(simpleCosmicBONSeeds)
simpleCosmicBONSeedsBottom.PositiveYOnly = False
simpleCosmicBONSeedsBottom.NegativeYOnly = True
simpleCosmicBONSeedsBottom.ClusterCheckPSet.ClusterCollectionLabel = cms.InputTag("siStripClustersBottom")
simpleCosmicBONSeedsBottom.TripletsSrc = "simpleCosmicBONSeedingLayersBottom"
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
ckfTrackCandidatesP5Bottom = copy.deepcopy(ckfTrackCandidatesP5)
ckfTrackCandidatesP5Bottom.TrajectoryBuilderPSet.refToPSet_ = 'GroupedCkfTrajectoryBuilderP5Bottom'
ckfTrackCandidatesP5Bottom.NavigationSchool   = 'CosmicNavigationSchool'
ckfTrackCandidatesP5Bottom.src       = 'combinedP5SeedsForCTFBottom' #ok for 32X
#ckfTrackCandidatesP5Bottom.SeedProducer       = 'combinedP5SeedsForCTFBottom' #ok for 22X
ckfTrackCandidatesP5Bottom.useHitsSplitting = True
ctfWithMaterialTracksP5Bottom = copy.deepcopy(ctfWithMaterialTracksCosmics)
ctfWithMaterialTracksP5Bottom.src    = 'ckfTrackCandidatesP5Bottom'
ctfWithMaterialTracksP5Bottom.Fitter = 'FittingSmootherRKP5'
ctfWithMaterialTracksP5Bottom.clusterRemovalInfo = "topBottomClusterInfoProducerBottom"
ctftracksP5Bottom = cms.Sequence(combinatorialcosmicseedingtripletsP5Bottom*combinatorialcosmicseedingpairsTOBP5Bottom*
                                 combinatorialcosmicseedingpairsTECposP5Bottom*combinatorialcosmicseedingpairsTECnegP5Bottom*
                                 combinatorialcosmicseedfinderP5Bottom*simpleCosmicBONSeedingLayersBottom*simpleCosmicBONSeedsBottom*
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


#TOP SEQUENCE
# (SK) keep rstracks commented out in case of resurrection
tracksP5Top = cms.Sequence(ctftracksP5Top+cosmictracksP5Top)
#BOTTOM SEQUENCE
# (SK) keep rstracks commented out in case of resurrection
tracksP5Bottom = cms.Sequence(ctftracksP5Bottom+cosmictracksP5Bottom)
