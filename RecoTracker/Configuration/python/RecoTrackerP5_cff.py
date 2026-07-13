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
from RecoTracker.PixelLowPtUtilities.siPixelClusterShapeCache_cfi import *

#chi2 set to 40!!
# CTF
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
from RecoTracker.SpecialSeedGenerators.CombinatorialSeedGeneratorForCosmicsP5_cff import *
from RecoTracker.SpecialSeedGenerators.SimpleCosmicBONSeeder_cff import *
from RecoTracker.SpecialSeedGenerators.CosmicGridTripletSeeder_cff import * 
from RecoTracker.TkSeedGenerator.GlobalCombinedSeeds_cff import *
combinedP5SeedsForCTF = RecoTracker.TkSeedGenerator.GlobalCombinedSeeds_cfi.globalCombinedSeeds.clone(
    seedCollections   = ['combinatorialcosmicseedfinderP5',
	                 'simpleCosmicBONSeeds']
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



ctfSeedsP5Task = cms.Task(combinatorialcosmicseedinglayersP5Task,
                                  simpleCosmicBONSeeds)

phase2SeedsP5Task = cms.Task(cosmicGridTripletSeeds)

# in phase 2, we run the CTF on top of the grid seeder. 
phase2_tracker.toReplaceWith(ctfSeedsP5Task,phase2SeedsP5Task)
phase2_tracker.toModify(combinedP5SeedsForCTF,seedCollections   = ['cosmicGridTripletSeeds'])

ctftracksP5Task = cms.Task( ctfSeedsP5Task,
                            combinedP5SeedsForCTF,
                            ckfTrackCandidatesP5, # CKF built from seeds
                            ctfWithMaterialTracksCosmics, # these are TrackCandidatesP5 + CKF fit 
                            ctfWithMaterialTracksP5,  # This is adding track selection on ctfWithMaterialTracksCosmics
                            ckfTrackCandidatesP5LHCNavigation,    # this is the CKF candidates using the collision-style navigation
                            ctfWithMaterialTracksP5LHCNavigation  # CKF candidates  using the collision-style navigation
                        ) 

# Copy of CTF on top of the grid seeder, used within Phase-1 for side-by-side validation 

combinedP5SeedsForGrid = combinedP5SeedsForCTF.clone(seedCollections   = ['cosmicGridTripletSeeds'])
gridTrackCandidatesP5 = ckfTrackCandidatesP5.clone(src = "combinedP5SeedsForGrid")
gridWithMaterialTracksCosmics = ctfWithMaterialTracksCosmics.clone(src = "gridTrackCandidatesP5")
gridWithMaterialTracksP5 = ctfWithMaterialTracksP5.clone(src = "gridWithMaterialTracksCosmics")
gridTrackCandidatesP5LHCNavigation    = gridTrackCandidatesP5.clone(NavigationSchool = 'SimpleNavigationSchool')
gridWithMaterialTracksP5LHCNavigation = gridWithMaterialTracksCosmics.clone(src = "gridTrackCandidatesP5LHCNavigation")

gridtracksP5Task = cms.Task(       
                            cosmicGridTripletSeeds,
                            combinedP5SeedsForGrid,
                            gridTrackCandidatesP5, # CKF built from seeds
                            gridWithMaterialTracksCosmics, # these are TrackCandidatesP5 + CKF fit 
                            gridWithMaterialTracksP5,  # This is adding track selection on ctfWithMaterialTracksCosmics
                            gridTrackCandidatesP5LHCNavigation,    # this is the CKF candidates using the collision-style navigation
                            gridWithMaterialTracksP5LHCNavigation # CKF candidates  using the collision-style navigation
                        ) 

ctftracksP5 = cms.Sequence(ctftracksP5Task)
gridtracksP5 = cms.Sequence(gridtracksP5Task)

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
                            gridtracksP5Task,
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
