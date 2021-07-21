import FWCore.ParameterSet.Config as cms

#
# Tracking configuration file fragment for P5 cosmic running
#
from RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cff import *
# TTRHBuilders
from RecoTracker.TransientTrackingRecHit.TTRHBuilders_cff import *
# COSMIC TRACK FINDER
from RecoTracker.SpecialSeedGenerators.CosmicSeedP5Pairs_cff import *
from RecoTracker.SingleTrackPattern.CosmicTrackFinderP5_cff import *
# Final Track Selector for CosmicTF
from RecoTracker.FinalTrackSelectors.CosmicTFFinalTrackSelectorP5_cff import *
from RecoPixelVertexing.PixelLowPtUtilities.siPixelClusterShapeCache_cfi import *

#chi2 set to 40!!
# CTF
from RecoTracker.SpecialSeedGenerators.CombinatorialSeedGeneratorForCosmicsP5_cff import *
from RecoTracker.SpecialSeedGenerators.SimpleCosmicBONSeeder_cff import *
from RecoTracker.TkSeedGenerator.GlobalCombinedSeeds_cff import *
combinedP5SeedsForCTF = RecoTracker.TkSeedGenerator.GlobalCombinedSeeds_cfi.globalCombinedSeeds.clone(
    seedCollections   = ['combinatorialcosmicseedfinderP5',
	                 'simpleCosmicBONSeeds'],
    #backward compatibility 2.2/3.1
    PairCollection    = cms.InputTag('combinatorialcosmicseedfinderP5'),
    TripletCollection = cms.InputTag('simpleCosmicBONSeeds')
)

from RecoTracker.CkfPattern.CkfTrackCandidatesP5_cff import *
ckfTrackCandidatesP5.src = 'combinedP5SeedsForCTF'
#backward compatibility 2.2/3.1

#import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
from RecoTracker.TrackProducer.CTFFinalFitWithMaterialP5_cff import *
# Final Track Selector for CTF
from RecoTracker.FinalTrackSelectors.CTFFinalTrackSelectorP5_cff import *

# ROACH SEARCH
# Final Track Selector for RS
#from RecoTracker.FinalTrackSelectors.RSFinalTrackSelectorP5_cff import *

# TRACK INFO

ckfTrackCandidatesP5LHCNavigation    = ckfTrackCandidatesP5.clone(NavigationSchool = 'SimpleNavigationSchool')
ctfWithMaterialTracksP5LHCNavigation = ctfWithMaterialTracksCosmics.clone(src = "ckfTrackCandidatesP5LHCNavigation")

ctftracksP5Task = cms.Task(combinatorialcosmicseedinglayersP5Task,
                                  combinatorialcosmicseedfinderP5,
                                  simpleCosmicBONSeedingLayers,
                                  simpleCosmicBONSeeds,
                                  combinedP5SeedsForCTF,
                                  ckfTrackCandidatesP5,
                                  ctfWithMaterialTracksCosmics,
                                  ctfWithMaterialTracksP5,
                                  ckfTrackCandidatesP5LHCNavigation,
                                  ctfWithMaterialTracksP5LHCNavigation)
ctftracksP5 = cms.Sequence(ctftracksP5Task)

from RecoTracker.FinalTrackSelectors.cosmicTrackSplitter_cfi import *
cosmicTrackSplitting = RecoTracker.FinalTrackSelectors.cosmicTrackSplitter_cfi.cosmicTrackSplitter.clone(
    tjTkAssociationMapTag = 'cosmictrackfinderCosmics',
    tracks = 'cosmictrackfinderCosmics'
)
splittedTracksP5 = cosmictrackfinderCosmics.clone(src = "cosmicTrackSplitting")

cosmictracksP5Task = cms.Task(cosmicseedfinderP5,
                              cosmicCandidateFinderP5,
                              cosmictrackfinderCosmics,
                              cosmictrackfinderP5,
                              cosmicTrackSplitting,
                              splittedTracksP5)

cosmictracksP5 = cms.Sequence(cosmictracksP5Task)

#Top/Bottom tracks NEW
from RecoTracker.Configuration.RecoTrackerTopBottom_cff import *
trackerCosmics_TopBotTask = cms.Task(trackerlocalrecoTopTask,
                                            tracksP5TopTask,
                                            trackerlocalrecoBottomTask,
                                            tracksP5BottomTask)
trackerCosmics_TopBot = cms.Sequence(trackerCosmics_TopBotTask)
#dEdX reconstruction
from RecoTracker.DeDx.dedxEstimators_Cosmics_cff import *
# (SK) keep rstracks commented out in case of resurrection
tracksP5Task = cms.Task(cosmictracksP5Task,
                            ctftracksP5Task,
                            doAllCosmicdEdXEstimatorsTask,
                            siPixelClusterShapeCache)
tracksP5 = cms.Sequence(tracksP5Task)
tracksP5_wodEdX = tracksP5.copy()
tracksP5_wodEdX.remove(doAllCosmicdEdXEstimators)

# explicitely switch on hit splitting
ckfTrackCandidatesP5.useHitsSplitting = True

# REGIONAL RECONSTRUCTION
from RecoTracker.Configuration.RecoTrackerNotStandard_cff import *
regionalCosmicTrackerSeeds.RegionInJetsCheckPSet = cms.PSet( doJetsExclusionCheck   = cms.bool( False ) )

# CDC Reconstruction
from RecoTracker.SpecialSeedGenerators.cosmicDC_cff import *
