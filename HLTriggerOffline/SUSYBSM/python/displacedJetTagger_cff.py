import FWCore.ParameterSet.Config as cms

# Tag jets with few prompt tracks.
# This is based on the displaced jet trigger path, but using offline quantities.
# It uses the b tag software framework.

# If you configure the following to use HLT objects as input, you will need to ensure that they
# were stored in the input file, and configure CMSSW to skip events where these products are
# not present (because the HLT path did not fully run).


displacedJetAssociator = cms.EDProducer( "JetTracksAssociatorAtVertex",

    jets = cms.InputTag( "ak5CaloJets" ),
# or use the same jets as used in the trigger.
#    jets = cms.InputTag( "hltAntiKT5L2L3CorrCaloJetsPt60Eta2V2"),
                                         
# use offline tracks
    tracks = cms.InputTag( "generalTracks" ),
# or use same tracks as used in trigger
#    tracks = cms.InputTag( "hltPixelTracks24cm" ),
#    tracks = cms.InputTag( "hltDisplacedHT250RegionalCtfWithMaterialTracksV2" ),
# These are not used in HLT, but available in RECO.
#    tracks = cms.InputTag( "pixelTracks" ),
    coneSize = cms.double( 0.5 )
)

from RecoBTag.ImpactParameter.impactParameter_cff import *
displacedJetTagInfos = impactParameterTagInfos.clone (
    jetTracks = cms.InputTag( "displacedJetAssociator" ),
    primaryVertex = cms.InputTag("offlinePrimaryVertices"),
# or use same priamry vertex as used in trigger
#    primaryVertex = cms.InputTag( "hltPixelVertices24cm" ),    
# These are not the ones used in HLT, but are available in RECO.
#    primaryVertex = cms.InputTag( "pixelVertices" ),    
    computeProbabilities = cms.bool( False ),
    computeGhostTrack = cms.bool( False ),
    maximumTransverseImpactParameter = cms.double( 0.1 ),
    maximumLongitudinalImpactParameter = cms.double( 0.1 ),
    minimumNumberOfPixelHits = cms.int32( 2 ),
# The following are needed if running on pixel tracks
#    minimumNumberOfHits = cms.int32( 3 ),
#    maximumChiSquared = cms.double( 5.0 )
# Whereas the following are needed if running on pixel + strip tracks.
    minimumNumberOfHits = cms.int32( 8 ),
    maximumChiSquared = cms.double( 20.0 )
)

hltESPPromptTrackCounting = cms.ESProducer("PromptTrackCountingESProducer",
    impactParameterType = cms.int32(0), ## 0 = 3D, 1 = 2D

    maximumDistanceToJetAxis = cms.double(999999.0),
    deltaR = cms.double(-1.0), ## use cut from JTA
    deltaRmin = cms.double( 0.0 ),
                                           
    maximumDecayLength = cms.double(999999.0),
    maxImpactParameter = cms.double( 0.03 ),                                           
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
