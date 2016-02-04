import FWCore.ParameterSet.Config as cms

ESTrivialConditionRetriever = cms.ESSource("ESTrivialConditionRetriever",
    #       Values to get correct noise on RecHit amplitude using 3+5 weights
    ESpedRMS = cms.untracked.double(1.26),
    weightsForTB = cms.untracked.bool(True),
    producedESPedestals = cms.untracked.bool(True),
    #       If set true reading optimized weights (3+5 weights) from file 
    getWeightsFromFile = cms.untracked.bool(False),
    producedESWeights = cms.untracked.bool(True),
    producedESIntercalibConstants = cms.untracked.bool(True),
    producedESADCToGeVConstant = cms.untracked.bool(True)
)


