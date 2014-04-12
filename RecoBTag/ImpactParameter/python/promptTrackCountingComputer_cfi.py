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
    deltaR = cms.double(-1.0), ## maximum deltaR of track to jet. If -ve just use cut from JTA
    deltaRmin = cms.double(0.0), ## minimum deltaR of track to jet.                                     

    maximumDecayLength = cms.double(999999.0),
    # These cuts on absolute impact parameter and its significance
    maxImpactParameter    = cms.double(0.1),
    maxImpactParameterSig = cms.double(999999.0),
    trackQualityClass = cms.string("any"),

    # This parameter is not used. 
    nthTrack = cms.int32(-1)                                    
)


