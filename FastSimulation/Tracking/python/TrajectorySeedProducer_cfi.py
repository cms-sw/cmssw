import FWCore.ParameterSet.Config as cms

trajectorySeedProducer = cms.EDProducer("TrajectorySeedProducer",
    # the name of the output seeds
    outputSeedCollectionName = cms.string("seeds"),
    # The smallest number of layer crossed to create a track candidate
    minRecHits = cms.uint32(5),
    skipSimTrackIdTags = cms.untracked.VInputTag(),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    originHalfLength = cms.double(15.9),
    # The number of hits needed to make a seed
    numberOfHits = cms.uint32(2),
    zVertexConstraint = cms.double(-1.0),
    originRadius = cms.double(0.2),
    # Inputs: tracker rechits, beam spot position.
    HitProducer = cms.InputTag("siTrackerGaussianSmearingRecHits","TrackerGSMatchedRecHits"),
    originpTMin = cms.double(1.0),
    #this skips the test to have two seeds compatible with the PV
    #Note: if no PV is set, BeamSpot is used.
    skipPVCompatibility = cms.bool(False),
    # The primary vertex collection
    primaryVertex = cms.InputTag("none"),
    
    # The smallest pT (true, in GeV/c) to create a track candidate 
    pTMin = cms.double(0.9),
    
    maxZ0 = cms.double(30.0),
    # The seed cuts for compatibility with originating from the beam axis.
    seedCleaning = cms.bool(True),
    # The smallest d0 and z0 (true, in cm) to create a track candidate
    maxD0 = cms.double(1.0),                     
                                     
    layerList = cms.vstring('BPix1+BPix2', 'BPix1+BPix3', 'BPix2+BPix3',
                            'BPix1+FPix1_pos', 'BPix1+FPix1_neg',
                            'BPix2+FPix1_pos', 'BPix2+FPix1_neg',
                            'FPix1_pos+FPix2_pos', 'FPix1_neg+FPix2_neg'),
)

