import FWCore.ParameterSet.Config as cms

# Tag jets with few prompt tracks.
# This is based on the displaced jet trigger path, but using offline quantities.
# It uses the b tag software framework.

displacedJetAssociator = cms.EDProducer( "JetTracksAssociatorAtVertex",
    jets = cms.InputTag( "ak5CaloJets" ),
# use offline tracks
    tracks = cms.InputTag( "generalTracks" ),
# or use same tracks as used in trigger
# (only works if you stored them in input file and skip events where this product
# is missing).
#    tracks = cms.InputTag( "hltPixelTracks" ),
#    tracks = cms.InputTag( "hltDisplacedHT240RegionalCtfWithMaterialTracks" ),
    coneSize = cms.double( 0.5 )
)

from RecoBTag.ImpactParameter.impactParameter_cff import *
displacedJetTagInfos = impactParameterTagInfos.clone (
    jetTracks = cms.InputTag( "displacedJetAssociator" ),
    computeProbabilities = cms.bool( False ),
    computeGhostTrack = cms.bool( False ),
#    maximumChiSquared = cms.double( 20.0 ),
    maximumTransverseImpactParameter = cms.double( 0.2 ),
    maximumLongitudinalImpactParameter = cms.double( 0.2 ),
# The following are needed only if running on L25 hltPixelTracks
    minimumNumberOfPixelHits = cms.int32( 2 ),
    minimumNumberOfHits = cms.int32( 3 ),
    maximumChiSquared = cms.double( 5.0 ),
)

hltESPPromptTrackCounting = cms.ESProducer("PromptTrackCountingESProducer",
    impactParameterType = cms.int32(0), ## 0 = 3D, 1 = 2D

    maximumDistanceToJetAxis = cms.double(999999.0),
    deltaR = cms.double(-1.0), ## use cut from JTA

    maximumDecayLength = cms.double(999999.0),
    # Warning, this cuts on absolute impact parameter significance
    maxImpactParameterSig = cms.double(999999.0),
    trackQualityClass = cms.string("any"),
#    trackQualityClass = cms.string("goodIterative"),

    # This parameter is not used. 
    nthTrack = cms.int32(-1)                                    
)

displacedJetTags = cms.EDProducer( "JetTagProducer",
    jetTagComputer = cms.string( "hltESPPromptTrackCounting" ),
    tagInfos = cms.VInputTag( 'displacedJetTagInfos' )
)

displacedJetSequence = cms.Sequence(displacedJetAssociator * displacedJetTagInfos * displacedJetTags)
