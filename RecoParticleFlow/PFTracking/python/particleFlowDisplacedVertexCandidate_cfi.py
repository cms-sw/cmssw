import FWCore.ParameterSet.Config as cms

particleFlowDisplacedVertexCandidate = cms.EDProducer("PFDisplacedVertexCandidateProducer",



    trackCollection  = cms.InputTag("generalTracks"),

    # verbosity 
    verbose = cms.untracked.bool(False),
                                   
    # Debug flag
    debug = cms.untracked.bool(False),

    # maximum dca distance for two tracks to be linked                                   
    dcaCut = cms.double(0.5),

    # minimum distance of secondary vertex with respect to the primary
    primaryVertexCut = cms.double(4.0),

    # maximum distance between the DCA Point and the inner hit of the track
    dcaPInnerHitCut = cms.double(40.0)
                                   
)


