import FWCore.ParameterSet.Config as cms

trajectorySeedProducer = cms.EDProducer("TrajectorySeedProducer",
    thirdHitSubDetectors = cms.vuint32(),
    # The smallest number of layer crossed to create a track candidate
    minRecHits = cms.vuint32(5),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    originHalfLength = cms.vdouble(15.9),
    # The number of hits needed to make a seed
    numberOfHits = cms.vuint32(2),
    zVertexConstraint = cms.vdouble(-1.0),
    originRadius = cms.vdouble(0.2),
    # Inputs: tracker rechits, beam spot position.
    HitProducer = cms.InputTag("siTrackerGaussianSmearingRecHits","TrackerGSMatchedRecHits"),
    originpTMin = cms.vdouble(1.0),
    firstHitSubDetectors = cms.vuint32(1, 2),
    # The primary vertex collection
    primaryVertices = cms.VInputTag(cms.InputTag("none")),
    secondHitSubDetectorNumber = cms.vuint32(2),
    # The smallest pT (true, in GeV/c) to create a track candidate 
    pTMin = cms.vdouble(0.9),
    seedingAlgo = cms.vstring('GlobalPixel'),
    maxZ0 = cms.vdouble(30.0),
    # The seed cuts for compatibility with originating from the beam axis.
    seedCleaning = cms.bool(True),
    # The smallest d0 and z0 (true, in cm) to create a track candidate
    maxD0 = cms.vdouble(1.0),
    thirdHitSubDetectorNumber = cms.vuint32(0),
    secondHitSubDetectors = cms.vuint32(1, 2),
    # The possible subdetectors for the first, second and third  hit
    # No seed with more than three hits are foreseen, but the code can certainly be 
    # modified to include this possibility.
    # 1 = PXB, 2 = PXD, 3 = TIB, 4 = TID, 5 = TOB, 6 = TEC 
    firstHitSubDetectorNumber = cms.vuint32(2)
)


