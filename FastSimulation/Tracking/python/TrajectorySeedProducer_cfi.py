import FWCore.ParameterSet.Config as cms

from RecoTracker.TkTrackingRegions.GlobalTrackingRegionFromBeamSpot_cfi import *
from RecoTracker.TkSeedGenerator.SeedFromConsecutiveHitsCreator_cfi import *

trajectorySeedProducer = cms.EDProducer("TrajectorySeedProducer",
                                        SeedCreatorPSet = cms.PSet(SeedFromConsecutiveHitsCreator.clone(TTRHBuilder = cms.string("WithoutRefit"))),
                                        # minimum number of layer crossed (with hits on them) by the simtrack
                                        minLayersCrossed = cms.uint32(0),
                                        
                                        #if empty, BS compatibility is skipped
                                        beamSpot = cms.InputTag("offlineBeamSpot"),
                                        #if empty, PV compatibility is skipped
                                        primaryVertex = cms.InputTag(""),
                                        
                                        nSigmaZ = cms.double(-1),
                                        originHalfLength= cms.double(-1),
                                        
                                        originRadius = cms.double(-1),
                                        ptMin = cms.double(-1),
                                        
                                        # Inputs: tracker rechits, beam spot position.
                                        recHits = cms.InputTag("siTrackerGaussianSmearingRecHits"),
                                        #hitMasks = cms.InputTag("hitMasks"),
                                        #hitCombinationMasks = cms.InputTag("hitCombinationMasks"),
                                        
                                        layerList = cms.vstring(),
                                        )

