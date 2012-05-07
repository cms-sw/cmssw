import FWCore.ParameterSet.Config as cms

# This computer counts the number of PROMPT tracks in a jet.
# i.e. The tagging variable it calculates is equal to this number.
# Its main use it for exotica physics, not b tagging.
# It counts tracks with impact parameter significance less than some cut.
# If you also wish to apply a cut on the maximum allowed impact parameter,
# you can do this in the TagInfoProducer.

promptTrackCounting = cms.ESProducer("PromptTrackCountingESProducer",
    impactParameterType = cms.int32(0), ## 0 = 3D, 1 = 2D

    maximumDistanceToJetAxis = cms.double(999999.0),
    deltaR = cms.double(-1.0), ## use cut from JTA

    maximumDecayLength = cms.double(999999.0),
    # Warning, this cuts on absolute impact parameter significance
    maxImpactParameterSig = cms.double(999999.0),
    trackQualityClass = cms.string("any"),

    # This parameter is not used. 
    nthTrack = cms.int32(-1)                                    
)


