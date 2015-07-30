import FWCore.ParameterSet.Config as cms

trajectorySeedProducer = cms.EDProducer("TrajectorySeedProducer",
                                        
    simTrackSelection = cms.PSet(
         # The smallest pT (in GeV) to create a track candidate 
         pTMin = cms.double(-1),
         # skip SimTracks processed in previous iterations
         #skipSimTrackIds = cms.VInputTag(),
         maxZ0 = cms.double(-1),
         maxD0 = cms.double(-1),
    ),
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

