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
combinedP5SeedsForCTF.PairCollection = 'combinatorialcosmicseedfinderP5'
combinedP5SeedsForCTF.TripletCollection = 'simpleCosmicBONSeeds'

from RecoTracker.CkfPattern.CkfTrackCandidatesP5_cff import *
ckfTrackCandidatesP5.SeedProducer = 'combinedP5SeedsForCTF'

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
cosmictracksP5 = cms.Sequence(cosmicseedfinderP5*cosmictrackfinderP5)
#sequence tracksP5 = {cosmictracksP5, ctftracksP5, rstracksP5, trackinfoP5}
tracksP5 = cms.Sequence(cosmictracksP5*ctftracksP5*rstracksP5)
# explicitely switch on hit splitting
ckfTrackCandidatesP5.useHitsSplitting = True
rsTrackCandidatesP5.SplitMatchedHits = True

