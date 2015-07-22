import FWCore.ParameterSet.Config as cms

particleFlowDisplacedVertexCandidate = cms.EDProducer("PFDisplacedVertexCandidateProducer",


    # The track collection use for the fitting. May be any collection.
    # The only condition is that it shall contain the hit pattern information
    trackCollection  = cms.InputTag("generalTracks"),

    # verbosity 
    verbose = cms.untracked.bool(False),
                                   
    # Debug flag
    debug = cms.untracked.bool(False),

    # maximum dca distance for two tracks to be linked                                   
    dcaCut = cms.double(0.5),

    # minimum distance of secondary vertex with respect to the primary
    primaryVertexCut = cms.double(1.8),

    # maximum distance between the DCA Point and the inner hit of the track
    # not used for the moment
    dcaPInnerHitCut = cms.double(1000.0),

    # Primary vertex information used for dxy calculation
    mainVertexLabel = cms.InputTag("offlinePrimaryVertices", ""),
    offlineBeamSpotLabel = cms.InputTag("offlineBeamSpot", ""),

    # Tracks preselection to reduce the combinatorics in PFDisplacedVertexCandidates
    # this cuts are repeated then in a smarter way in the PFDisplacedVertexFinder
    # be sure you are consistent between them.
    tracksSelectorParameters = cms.PSet(
    # selection parameters for secondary tracks
        nChi2_max = cms.double(5.),
        pt_min = cms.double(.2),
    # if the tracks is not a good candidate to be a secondary (dxy cut) restrict in minimal pt
    # this cut reduce drastically the combinatorics. It is very useful to reduce the
    # PFDisplacedVertex timing
        pt_min_prim = cms.double(.8),
        dxy = cms.double(.2),
        
    )                                                     
                                   
)


