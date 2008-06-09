import FWCore.ParameterSet.Config as cms

#
# Tracking configuration file fragment for P5 cosmic running
#
# COSMIC TRACK FINDER
from RecoTracker.SpecialSeedGenerators.CosmicSeedP5Pairs_cff import *
from RecoTracker.SingleTrackPattern.CosmicTrackFinderP5_cff import *
#chi2 set to 40!!
# CTF
from RecoTracker.SpecialSeedGenerators.CombinatorialSeedGeneratorForCosmicsP5_cff import *
from RecoTracker.CkfPattern.CkfTrackCandidatesP5_cff import *
from RecoTracker.TrackProducer.CTFFinalFitWithMaterialP5_cff import *
# ROACH SEARCH
from RecoTracker.RoadSearchSeedFinder.RoadSearchSeedsP5_cff import *
from RecoTracker.RoadSearchCloudMaker.RoadSearchCloudsP5_cff import *
from RecoTracker.RoadSearchTrackCandidateMaker.RoadSearchTrackCandidatesP5_cff import *
from RecoTracker.TrackProducer.RSFinalFitWithMaterialP5_cff import *
# TRACK INFO
from AnalysisAlgos.TrackInfoProducer.TrackInfoProducerP5_cff import *
ctftracksP5 = cms.Sequence(combinatorialcosmicseedfinderP5*ckfTrackCandidatesP5*ctfWithMaterialTracksP5)
rstracksP5 = cms.Sequence(roadSearchSeedsP5*roadSearchCloudsP5*rsTrackCandidatesP5*rsWithMaterialTracksP5)
cosmictracksP5 = cms.Sequence(cosmicseedfinderP5*cosmictrackfinderP5)
tracksP5 = cms.Sequence(cosmictracksP5*ctftracksP5*rstracksP5*trackinfoP5)

