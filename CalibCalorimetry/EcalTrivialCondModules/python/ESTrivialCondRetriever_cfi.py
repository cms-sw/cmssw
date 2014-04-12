import FWCore.ParameterSet.Config as cms

ESTrivialConditionRetriever = cms.ESSource("ESTrivialConditionRetriever",
    producedChannelStatus = cms.untracked.bool(True),
    #       Values to get correct noise on RecHit amplitude using 3+5 weights
    ESpedRMS = cms.untracked.double(1.0),
    weightsForTB = cms.untracked.bool(False),
    # channel status
    channelStatusFile = cms.untracked.string(''),
    producedESPedestals = cms.untracked.bool(True),
    getWeightsFromFile = cms.untracked.bool(True),
    intercalibConstantsFile = cms.untracked.string(''),
    producedESWeights = cms.untracked.bool(True),
    producedESIntercalibConstants = cms.untracked.bool(True),
    producedESADCToGeVConstant = cms.untracked.bool(True),
    adcToGeVESLowConstant = cms.untracked.double(1.0),
    adcToGeVESHighConstant = cms.untracked.double(0.5),

)
