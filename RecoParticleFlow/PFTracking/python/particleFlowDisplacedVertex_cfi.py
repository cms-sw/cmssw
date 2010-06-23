import FWCore.ParameterSet.Config as cms

particleFlowDisplacedVertex = cms.EDProducer("PFDisplacedVertexProducer",



    trackCollection  = cms.InputTag("generalTracks"),

    vertexCandidates = cms.InputTag("particleFlowDisplacedVertexCandidate"),                                         

    # verbosity 
    verbose = cms.untracked.bool(False),
                                   
    # Debug flag
    debug = cms.untracked.bool(False),

    # maximum transverse distance between two points to be used in Seed                                   
    transvSize = cms.double(1.0),

    # maximum longitudinal distance between two points to be used in Seed                                   
    longSize = cms.double(5),

    # minimal radius below which we do not reconstruct interactions
    # typically the position of the first Pixel layer

    primaryVertexCut = cms.double(1.5),

    # radius below which we don't wamt to reconstruct displaced
    # vertices
    tobCut = cms.double(100),

    # z below which we don't want to reconstruct displaced
    # vertices
    tecCut = cms.double(220),

    # the minimal accepted weight for the tracks calculated in the 
    # adaptive vertex fitter to be associated to the displaced vertex
    # this correspond to the sigmacut of 6
    minAdaptWeight = cms.double(0.5)
                                   
)


