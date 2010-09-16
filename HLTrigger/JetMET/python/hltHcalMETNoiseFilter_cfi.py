import FWCore.ParameterSet.Config as cms
 
hltHcalMETNoiseFilter = cms.EDFilter(
    "HLTHcalMETNoiseFilter",

    # collections to get
    HcalNoiseRBXCollection = cms.InputTag("hltHcalNoise"),
    
    # set to 0 if you want to accept all events
    severity = cms.int32(1),

    # if there are more than maxNumRBXs RBXs in the event, the event passes the trigger
    maxNumRBXs = cms.int32(2),
    
    # consider the top N=numRBXsToConsider RBXs by energy in the event
    # this number should be <= maxNumRBXs
    numRBXsToConsider = cms.int32(2),

    # require coincidence between the High-Level (EMF) filter and the other filters
    needEMFCoincidence = cms.bool(True),

    # cuts
    minRBXEnergy = cms.double(50.0),
    minRatio = cms.double(0.65),
    maxRatio = cms.double(0.98),
    minHPDHits = cms.int32(17),
    minRBXHits = cms.int32(999),
    minHPDNoOtherHits = cms.int32(10),
    minZeros = cms.int32(10),
    minLowEHitTime = cms.double(-9999.0),
    maxLowEHitTime = cms.double(9999.0),
    minHighEHitTime = cms.double(-9999.0),
    maxHighEHitTime = cms.double(9999.0),
    
    maxRBXEMF = cms.double(0.02),

    minRecHitE = cms.double(1.5),
    minLowHitE = cms.double(10.0),
    minHighHitE = cms.double(25.0),

    )
