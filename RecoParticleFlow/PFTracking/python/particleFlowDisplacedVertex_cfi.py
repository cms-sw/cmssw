import FWCore.ParameterSet.Config as cms

particleFlowDisplacedVertex = cms.EDProducer("PFDisplacedVertexProducer",



    trackCollection  = cms.InputTag("generalTracks"),

    vertexCandidates = cms.InputTag("particleFlowDisplacedVertexCandidate"),                                         

    # verbosity 
    verbose = cms.untracked.bool(True),
                                   
    # Debug flag
    debug = cms.untracked.bool(False),

    # maximum transverse distance between two points to be used in Seed                                   
    transvSize = cms.double(1.0),

    # maximum longitudinal distance between two points to be used in Seed                                   
    longSize = cms.double(5),

    # minimal radius below which we do not reconstruct interactions
    # typically the position of the first Pixel layer

    primaryVertexCut = cms.double(4),

    # radius at which no secondary tracks are availables
    # in the barrel.For the moment we exclude the TOB barrel
    # since 5-th track step starts the latest at first TOB
    # layer.
    tobCut = cms.double(56),

    # radius at which no secondary tracks are availables
    # in the endcaps.For the moment we exclude the TEC wheel.
    tecCut = cms.double(110),

    # the minimal accepted weight for the tracks calculated in the 
    # adaptive vertex fitter to be associated to the displaced vertex
    # 0.5 corresponds to the 3-sigma uncompatibility (to check)
    minAdaptWeight = cms.double(0.5)
                                   
)


