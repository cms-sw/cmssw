import FWCore.ParameterSet.Config as cms

#
# Tracker Tracking etc
#
# Seeds 
#include "RecoTracker/TkSeedGenerator/data/GlobalMixedSeeds.cff"
#include "RecoTracker/TkSeedGenerator/data/GlobalPixelSeeds.cff"                           
# Ckf
#include "RecoTracker/CkfPattern/data/CkfTrackCandidates.cff"
# DA CAMBIARE ?
from RecoEgamma.EgammaHLTProducers.l1IsoElectronsRegionalCkfTrackCandidates_cff import *
# Final Fit
#include "RecoTracker/TrackProducer/data/CTFFinalFitWithMaterial.cff"
from RecoEgamma.EgammaHLTProducers.l1IsoElectronsRegionalCTFFinalFitWithMaterial_cff import *
# RoadSearchSeedFinder
#include "RecoTracker/RoadSearchSeedFinder/data/RoadSearchSeeds.cff"
# RoadSearchCloudMaker
from RecoTracker.RoadSearchCloudMaker.RoadSearchClouds_cff import *
# RoadSearchCloudMaker
from RecoTracker.RoadSearchCloudCleaner.CleanRoadSearchClouds_cff import *
# RoadSearchTrackCandidateMaker
from RecoTracker.RoadSearchTrackCandidateMaker.RoadSearchTrackCandidates_cff import *
# RS track fit with material 
from RecoTracker.TrackProducer.RSFinalFitWithMaterial_cff import *
# act locally, not globally!  so regional pixel tracking, let's go:
from RecoEgamma.EgammaHLTProducers.l1IsoElectronsRegionalPixelSeedGenerator_cff import *
l1IsoElectronsRegionalRecoTracker = cms.Sequence(l1IsoElectronsRegionalPixelSeedGenerator*l1IsoElectronsRegionalCkfTrackCandidates*l1IsoElectronsRegionalCTFFinalFitWithMaterial)

