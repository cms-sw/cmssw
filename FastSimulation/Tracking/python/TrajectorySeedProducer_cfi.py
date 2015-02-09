import FWCore.ParameterSet.Config as cms

trajectorySeedProducer = cms.EDProducer("TrajectorySeedProducer",
                                        
    simTrackSelection = cms.PSet(
         # The smallest pT (in GeV) to create a track candidate 
         pTMin = cms.double(0.9),
         # skip SimTracks processed in previous iterations
         skipSimTrackIdTags = cms.VInputTag(),
         maxZ0 = cms.double(30.0),
         maxD0 = cms.double(1.0),
    ),
    # the name of the output seeds
    outputSeedCollectionName = cms.string("seeds"),
    
    # The smallest number of layer crossed to create a track candidate
    minRecHits = cms.uint32(5),
    numberOfHits = cms.uint32(2),
    
    #if empty, BS compatibility is skipped
    beamSpot = cms.InputTag("offlineBeamSpot"),
    
    originHalfLength = cms.double(15.9),

    zVertexConstraint = cms.double(-1.0),
    originRadius = cms.double(0.2),
    # Inputs: tracker rechits, beam spot position.
    HitProducer = cms.InputTag("siTrackerGaussianSmearingRecHits","TrackerGSMatchedRecHits"),
    originpTMin = cms.double(1.0),

    #if empty, PV compatibility is skipped
    primaryVertex = cms.InputTag(),
    
    # The smallest d0 and z0 (true, in cm) to create a track candidate

                                     
    layerList = cms.vstring('BPix1+BPix2', 'BPix1+BPix3', 'BPix2+BPix3',
                            'BPix1+FPix1_pos', 'BPix1+FPix1_neg',
                            'BPix2+FPix1_pos', 'BPix2+FPix1_neg',
                            'FPix1_pos+FPix2_pos', 'FPix1_neg+FPix2_neg'),
)

