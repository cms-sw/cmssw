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
#chi2 set to 40!!
# CTF
from RecoTracker.SpecialSeedGenerators.CombinatorialSeedGeneratorForCosmicsP5_cff import *
from RecoTracker.SpecialSeedGenerators.SimpleCosmicBONSeeder_cfi import *
from RecoTracker.TkSeedGenerator.GlobalCombinedSeeds_cff import *
combinedP5SeedsForCTF = RecoTracker.TkSeedGenerator.GlobalCombinedSeeds_cfi.globalCombinedSeeds.clone()
combinedP5SeedsForCTF.seedCollections = cms.VInputTag(
    cms.InputTag('combinatorialcosmicseedfinderP5'),
    cms.InputTag('simpleCosmicBONSeeds'),
)

from RecoTracker.CkfPattern.CkfTrackCandidatesP5_cff import *
ckfTrackCandidatesP5.src = cms.InputTag('combinedP5SeedsForCTF')

from RecoTracker.TrackProducer.CTFFinalFitWithMaterialP5_cff import *
# ROACH SEARCH
from RecoTracker.RoadSearchSeedFinder.RoadSearchSeedsP5_cff import *
from RecoTracker.RoadSearchCloudMaker.RoadSearchCloudsP5_cff import *
from RecoTracker.RoadSearchTrackCandidateMaker.RoadSearchTrackCandidatesP5_cff import *
from RecoTracker.TrackProducer.RSFinalFitWithMaterialP5_cff import *
# TRACK INFO
#include "AnalysisAlgos/TrackInfoProducer/data/TrackInfoProducerP5.cff"

ctftracksP5 = cms.Sequence(combinatorialcosmicseedfinderP5*simpleCosmicBONSeeds*combinedP5SeedsForCTF*
                           ckfTrackCandidatesP5*
                           ctfWithMaterialTracksP5)

rstracksP5 = cms.Sequence(roadSearchSeedsP5*roadSearchCloudsP5*rsTrackCandidatesP5*rsWithMaterialTracksP5)

from RecoTracker.FinalTrackSelectors.cosmicTrackSplitter_cfi import *
cosmicTrackSplitter.tjTkAssociationMapTag = 'cosmictrackfinderP5'
cosmicTrackSplitter.tracks = 'cosmictrackfinderP5'
splittedTracksP5 = ctfWithMaterialTracksP5.clone(src = cms.InputTag("cosmicTrackSplitter"))
    
cosmictracksP5 = cms.Sequence(cosmicseedfinderP5*cosmicCandidateFinderP5*cosmictrackfinderP5*cosmicTrackSplitter*splittedTracksP5)



#TOP tracking (RS, CTF, CoTF)
from RecoTracker.FinalTrackSelectors.TrackCandidateTopBottomHitFilter_cfi import *
import copy

combinatorialcosmicseedfinderP5Top = copy.deepcopy(combinatorialcosmicseedfinderP5)
simpleCosmicBONSeedsTop            = copy.deepcopy(simpleCosmicBONSeeds)
combinedP5SeedsForCTFTop           = copy.deepcopy(combinedP5SeedsForCTF)
ckfTrackCandidatesP5Top            = copy.deepcopy(ckfTrackCandidatesP5)
ctfWithMaterialTracksP5Top         = copy.deepcopy(ctfWithMaterialTracksP5)
filterCkfTop                       = copy.deepcopy(trackCandidateTopBottomHitFilter);
combinatorialcosmicseedfinderP5Top.SeedsFromPositiveY = True
combinatorialcosmicseedfinderP5Top.SeedsFromNegativeY = False
simpleCosmicBONSeedsTop.PositiveYOnly = True
simpleCosmicBONSeedsTop.NegativeYOnly = False
combinedP5SeedsForCTFTop.PairCollection    = 'combinatorialcosmicseedfinderP5Top'
combinedP5SeedsForCTFTop.TripletCollection = 'simpleCosmicBONSeedsTop'
ckfTrackCandidatesP5Top.NavigationSchool   = 'CosmicNavigationSchool'
ckfTrackCandidatesP5Top.SeedProducer       = 'combinedP5SeedsForCTFTop'
ckfTrackCandidatesP5Top.useHitsSplitting = True
filterCkfTop.Input = 'ckfTrackCandidatesP5Top'
filterCkfTop.SeedY = 1.
ctfWithMaterialTracksP5Top.src    = 'filterCkfTop'
ctfWithMaterialTracksP5Top.Fitter = 'FittingSmootherRKP5'
ctftracksP5Top = cms.Sequence(combinatorialcosmicseedfinderP5Top*simpleCosmicBONSeedsTop*combinedP5SeedsForCTFTop*
                              ckfTrackCandidatesP5Top*filterCkfTop*ctfWithMaterialTracksP5Top)

#COSMIC TOP
cosmicseedfinderP5Top       = copy.deepcopy(cosmicseedfinderP5)
cosmicCandidateFinderP5Top  = copy.deepcopy(cosmicCandidateFinderP5)
cosmictrackfinderP5Top      = copy.deepcopy(cosmictrackfinderP5)
filterCosmicTop             = copy.deepcopy(trackCandidateTopBottomHitFilter);
cosmicseedfinderP5Top.PositiveYOnly = True
cosmicseedfinderP5Top.NegativeYOnly = False
cosmicCandidateFinderP5Top.cosmicSeeds = 'cosmicseedfinderP5Top'
filterCosmicTop.Input = 'cosmicCandidateFinderP5Top'
filterCosmicTop.SeedY = 1.
cosmictrackfinderP5Top.src = 'filterCosmicTop'
cosmictracksP5Top = cms.Sequence(cosmicseedfinderP5Top*cosmicCandidateFinderP5Top*filterCosmicTop*cosmictrackfinderP5Top)

#RS TOP
roadSearchSeedsP5Top      = copy.deepcopy(roadSearchSeedsP5)
roadSearchCloudsP5Top     = copy.deepcopy(roadSearchCloudsP5)
rsTrackCandidatesP5Top    = copy.deepcopy(rsTrackCandidatesP5)
rsWithMaterialTracksP5Top = copy.deepcopy(rsWithMaterialTracksP5)
filterRSTop               = copy.deepcopy(trackCandidateTopBottomHitFilter);
roadSearchSeedsP5Top.AllPositiveOnly = True
roadSearchCloudsP5Top.SeedProducer = 'roadSearchSeedsP5Top'
rsTrackCandidatesP5Top.CloudProducer = 'roadSearchCloudsP5Top'
rsTrackCandidatesP5Top.SplitMatchedHits = True
filterRSTop.Input = 'rsTrackCandidatesP5Top'
filterRSTop.SeedY = 1.
rsWithMaterialTracksP5Top.src = 'filterRSTop'
rstracksP5Top = cms.Sequence(roadSearchSeedsP5Top*roadSearchCloudsP5Top*
                                     rsTrackCandidatesP5Top*filterRSTop*rsWithMaterialTracksP5Top)

#TOP SEQUENCE
tracksP5Top = cms.Sequence(ctftracksP5Top+cosmictracksP5Top+rstracksP5Top)


#Bottom tracking (RS, CTF, CoTF)
#CKF BOTTOM
combinatorialcosmicseedfinderP5Bottom = copy.deepcopy(combinatorialcosmicseedfinderP5)
simpleCosmicBONSeedsBottom = copy.deepcopy(simpleCosmicBONSeeds)
combinedP5SeedsForCTFBottom = copy.deepcopy(combinedP5SeedsForCTF)
ckfTrackCandidatesP5Bottom            = copy.deepcopy(ckfTrackCandidatesP5)
ctfWithMaterialTracksP5Bottom         = copy.deepcopy(ctfWithMaterialTracksP5)
filterCkfBottom                       = copy.deepcopy(trackCandidateTopBottomHitFilter);
combinatorialcosmicseedfinderP5Bottom.SeedsFromPositiveY = False
combinatorialcosmicseedfinderP5Bottom.SeedsFromNegativeY = True
combinatorialcosmicseedfinderP5Bottom.OrderedHitsFactoryPSets[0].PropagationDirection = cms.string('oppositeToMomentum')
combinatorialcosmicseedfinderP5Bottom.OrderedHitsFactoryPSets[1].PropagationDirection = cms.string('oppositeToMomentum')
combinatorialcosmicseedfinderP5Bottom.OrderedHitsFactoryPSets[2].PropagationDirection = cms.string('oppositeToMomentum')
combinatorialcosmicseedfinderP5Bottom.OrderedHitsFactoryPSets[3].PropagationDirection = cms.string('oppositeToMomentum')
combinatorialcosmicseedfinderP5Bottom.OrderedHitsFactoryPSets[4].PropagationDirection = cms.string('oppositeToMomentum')
combinatorialcosmicseedfinderP5Bottom.OrderedHitsFactoryPSets[5].PropagationDirection = cms.string('oppositeToMomentum')

simpleCosmicBONSeedsBottom.PositiveYOnly = False
simpleCosmicBONSeedsBottom.NegativeYOnly = True
combinedP5SeedsForCTFBottom.PairCollection = 'combinatorialcosmicseedfinderP5Bottom'
combinedP5SeedsForCTFBottom.TripletCollection = 'simpleCosmicBONSeedsBottom'
ckfTrackCandidatesP5Bottom.NavigationSchool = 'CosmicNavigationSchool'
ckfTrackCandidatesP5Bottom.SeedProducer = 'combinedP5SeedsForCTFBottom'
ckfTrackCandidatesP5Bottom.useHitsSplitting = True
filterCkfBottom.Input = 'ckfTrackCandidatesP5Bottom'
filterCkfBottom.SeedY = -1.
ctfWithMaterialTracksP5Bottom.src = 'filterCkfBottom'
ctfWithMaterialTracksP5Bottom.Fitter = 'FittingSmootherRKP5'
ctftracksP5Bottom = cms.Sequence(combinatorialcosmicseedfinderP5Bottom*simpleCosmicBONSeedsBottom*combinedP5SeedsForCTFBottom*
                                      ckfTrackCandidatesP5Bottom*filterCkfBottom*ctfWithMaterialTracksP5Bottom)

#COSMIC BOTTOM
cosmicseedfinderP5Bottom      = copy.deepcopy(cosmicseedfinderP5)
cosmicCandidateFinderP5Bottom = copy.deepcopy(cosmicCandidateFinderP5)
cosmictrackfinderP5Bottom     = copy.deepcopy(cosmictrackfinderP5)
filterCosmicBottom            = copy.deepcopy(trackCandidateTopBottomHitFilter);
cosmicseedfinderP5Bottom.PositiveYOnly = False
cosmicseedfinderP5Bottom.NegativeYOnly = True
cosmicCandidateFinderP5Bottom.cosmicSeeds = 'cosmicseedfinderP5Bottom'
filterCosmicBottom.Input = 'cosmicCandidateFinderP5Bottom'
filterCosmicBottom.SeedY = -1.
cosmictrackfinderP5Bottom.src = 'filterCosmicBottom'
cosmictracksP5Bottom = cms.Sequence(cosmicseedfinderP5Bottom*cosmicCandidateFinderP5Bottom*filterCosmicBottom*cosmictrackfinderP5Bottom)


#RS BOTTOM
roadSearchSeedsP5Bottom      = copy.deepcopy(roadSearchSeedsP5)
roadSearchCloudsP5Bottom     = copy.deepcopy(roadSearchCloudsP5)
rsTrackCandidatesP5Bottom    = copy.deepcopy(rsTrackCandidatesP5)
rsWithMaterialTracksP5Bottom = copy.deepcopy(rsWithMaterialTracksP5)
filterRSBottom               = copy.deepcopy(trackCandidateTopBottomHitFilter);
roadSearchSeedsP5Bottom.AllNegativeOnly = True
roadSearchCloudsP5Bottom.SeedProducer = 'roadSearchSeedsP5Bottom'
rsTrackCandidatesP5Bottom.CloudProducer = 'roadSearchCloudsP5Bottom'
rsTrackCandidatesP5Bottom.SplitMatchedHits = True
#rsTrackCandidatesP5Bottom.CosmicTrackMerging = True
filterRSBottom.Input = 'rsTrackCandidatesP5Bottom'
filterRSBottom.SeedY = -1.
rsWithMaterialTracksP5Bottom.src = 'filterRSBottom'
rstracksP5Bottom = cms.Sequence(roadSearchSeedsP5Bottom*roadSearchCloudsP5Bottom*
                                rsTrackCandidatesP5Bottom*filterRSBottom*rsWithMaterialTracksP5Bottom)

#BOTTOM SEQUENCE
tracksP5Bottom = cms.Sequence(ctftracksP5Bottom+cosmictracksP5Bottom+rstracksP5Bottom)

trackerCosmics_TopBot = cms.Sequence(tracksP5Top+tracksP5Bottom)



#sequence tracksP5 = {cosmictracksP5, ctftracksP5, rstracksP5, trackinfoP5}
tracksP5 = cms.Sequence(cosmictracksP5*ctftracksP5*rstracksP5*trackerCosmics_TopBot)
# explicitely switch on hit splitting
ckfTrackCandidatesP5.useHitsSplitting = True
rsTrackCandidatesP5.SplitMatchedHits = True


