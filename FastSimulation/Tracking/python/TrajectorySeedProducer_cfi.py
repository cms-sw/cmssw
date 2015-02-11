import FWCore.ParameterSet.Config as cms

trajectorySeedProducer = cms.EDProducer("TrajectorySeedProducer",
                                        
    simTrackSelection = cms.PSet(
         # The smallest pT (in GeV) to create a track candidate 
         pTMin = cms.double(0.9),
         # skip SimTracks processed in previous iterations
         skipSimTrackIds = cms.VInputTag(),
         maxZ0 = cms.double(30.0),
         maxD0 = cms.double(1.0),
    ),
    # minimum number of layer crossed (with hits on them) by the simtrack
    minLayersCrossed = cms.uint32(5),
    
    #if empty, BS compatibility is skipped
    beamSpot = cms.InputTag("offlineBeamSpot"),
    #if empty, PV compatibility is skipped
    primaryVertex = cms.InputTag(""),
    
    # seed producer will use min(nSigmaZ*sigmaZ(BS or PV),originHalfLength)
    nSigmaZ = cms.double(10000.0),
    originHalfLength= cms.double(15.9),
    
    originRadius = cms.double(0.2),
    originpTMin = cms.double(1.0),

    # Inputs: tracker rechits, beam spot position.
    recHits = cms.InputTag("siTrackerGaussianSmearingRecHits","TrackerGSMatchedRecHits"),

    layerList = cms.vstring(),
)

