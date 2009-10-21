import FWCore.ParameterSet.Config as cms

particleFlowBlock = cms.EDProducer("PFBlockProducer",

    # verbosity 
    verbose = cms.untracked.bool(False),
                                   

    # input clusters
    PFClustersECAL = cms.InputTag("particleFlowClusterECAL"),
    PFClustersHCAL = cms.InputTag("particleFlowClusterHCAL"),
    PFClustersHFEM = cms.InputTag("particleFlowClusterHFEM"),
    PFClustersHFHAD = cms.InputTag("particleFlowClusterHFHAD"),
    PFClustersPS = cms.InputTag("particleFlowClusterPS"),

    # input tracks
    GsfRecTracks = cms.InputTag("pfTrackElec"),
    ConvBremGsfRecTracks = cms.InputTag("pfTrackElec","Secondary"),
    useConvBremGsfTracks = cms.bool(True),                                     
    RecTracks = cms.InputTag("trackerDrivenElectronSeeds"),

    # input nuclear interactions 
    PFNuclear = cms.InputTag("pfNuclear"),
    useNuclear = cms.bool(False),

    # input muons
    RecMuons = cms.InputTag("muons"),

    # input conversions
    PFConversions = cms.InputTag("pfConversions"),
    useConversions = cms.bool(False),

    # input V0
    PFV0 = cms.InputTag("pfV0"),
    useV0 = cms.bool(False),

    # Track Quality Cut: Tracks are kept if DPt/Pt < sigma * Cut
    # and if nHit >= cut
    pf_DPtoverPt_Cut = cms.vdouble(-1.0,-1.0,-1.0,-1.0,-1.0),
    pf_NHit_Cut = cms.vuint32(3,3,3,3,3),
                                   
    # Run particle flow at HLT (hence no RecMuons, no GSF tracks)
    usePFatHLT = cms.bool(False),

    # Debug flag
    debug = cms.untracked.bool(False)
)


