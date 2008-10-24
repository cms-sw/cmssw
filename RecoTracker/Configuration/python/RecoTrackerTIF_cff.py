import FWCore.ParameterSet.Config as cms

#
# Tracker Tracking etc
#
# cosmic track finder seeding
from RecoTracker.SpecialSeedGenerators.CosmicSeedTIF_cff import *
# cosmic track finder pattern recognition and track fit
from RecoTracker.SingleTrackPattern.CosmicTrackFinderTIF_cff import *
# Seeds 
from RecoTracker.SpecialSeedGenerators.CombinatorialSeedGeneratorForCosmicsTIF_cff import *
# Ckf
from RecoTracker.CkfPattern.CkfTrackCandidatesTIF_cff import *
# Final Fit
from RecoTracker.TrackProducer.CTFFinalFitWithMaterialTIF_cff import *
# RoadSearchSeedFinder
from RecoTracker.RoadSearchSeedFinder.RoadSearchSeedsTIF_cff import *
# RoadSearchCloudMaker
from RecoTracker.RoadSearchCloudMaker.RoadSearchCloudsTIF_cff import *
# RoadSearchTrackCandidateMaker
from RecoTracker.RoadSearchTrackCandidateMaker.RoadSearchTrackCandidatesTIF_cff import *
# RS track fit with material 
from RecoTracker.TrackProducer.RSFinalFitWithMaterialTIF_cff import *
# include track info
#from AnalysisAlgos.TrackInfoProducer.TrackInfoProducerTIF_cff import *
ckftracksTIF = cms.Sequence(combinatorialcosmicseedfinderTIF*ckfTrackCandidatesTIF*ctfWithMaterialTracksTIF)
rstracksTIF = cms.Sequence(roadSearchSeedsTIF*roadSearchCloudsTIF*rsTrackCandidatesTIF*rsWithMaterialTracksTIF)
cosmictracksTIF = cms.Sequence(cosmicseedfinderTIF*cosmictrackfinderTIF)
#tracksTIF = cms.Sequence(cosmictracksTIF*ckftracksTIF*rstracksTIF*trackinfoTIF)
tracksTIF = cms.Sequence(cosmictracksTIF*ckftracksTIF*rstracksTIF)

