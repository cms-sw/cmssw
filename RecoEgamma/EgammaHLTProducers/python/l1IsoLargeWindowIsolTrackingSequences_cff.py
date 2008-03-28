import FWCore.ParameterSet.Config as cms

# Ckf
# DA CAMBIARE ?
from RecoEgamma.EgammaHLTProducers.l1IsoEgammaRegionalCkfTrackCandidates_cff import *
# Final Fit
#include "RecoTracker/TrackProducer/data/CTFFinalFitWithMaterial.cff"
from RecoEgamma.EgammaHLTProducers.l1IsoEgammaRegionalCTFFinalFitWithMaterial_cff import *
# RoadSearchCloudMaker
from RecoTracker.RoadSearchCloudMaker.RoadSearchClouds_cff import *
# RoadSearchCloudMaker
from RecoTracker.RoadSearchCloudCleaner.CleanRoadSearchClouds_cff import *
# RoadSearchTrackCandidateMaker
from RecoTracker.RoadSearchTrackCandidateMaker.RoadSearchTrackCandidates_cff import *
# RS track fit with material 
from RecoTracker.TrackProducer.RSFinalFitWithMaterial_cff import *
from RecoEgamma.EgammaHLTProducers.hltEgammaRegionalPixelSeedGenerator_cfi import *
import copy
from RecoEgamma.EgammaHLTProducers.hltEgammaRegionalPixelSeedGenerator_cfi import *
l1IsoEgammaRegionalPixelSeedGenerator = copy.deepcopy(hltEgammaRegionalPixelSeedGenerator)
#   string HitProducer = "siPixelRecHits"
#    double ptMin = 1.0
#    double vertexZ = 0.0
#    double originRadius = 0.2
#    double originHalfLength = 15.0
#    double deltaEtaRegion = .177 #1.0
#    double deltaPhiRegion = .177 #1.0
#    string TTRHBuilder = "WithTrackAngle"
#    InputTag candTag =  hltRecoEcalCandidate
l1IsoEgammaRegionalRecoTracker = cms.Sequence(l1IsoEgammaRegionalPixelSeedGenerator*l1IsoEgammaRegionalCkfTrackCandidates*l1IsoEgammaRegionalCTFFinalFitWithMaterial)
l1IsoEgammaRegionalPixelSeedGenerator.candTag = 'l1IsoRecoEcalCandidate'

