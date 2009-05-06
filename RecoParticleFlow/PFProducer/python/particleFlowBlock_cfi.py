import FWCore.ParameterSet.Config as cms

particleFlowBlock = cms.EDProducer("PFBlockProducer",

    # resolution maps
    pf_resolution_map_HCAL_eta = cms.string('RecoParticleFlow/PFBlockProducer/data/resmap_HCAL_eta.dat'),
    pf_resolution_map_HCAL_phi = cms.string('RecoParticleFlow/PFBlockProducer/data/resmap_HCAL_phi.dat'),
    pf_resolution_map_ECAL_eta = cms.string('RecoParticleFlow/PFBlockProducer/data/resmap_ECAL_eta.dat'),
    pf_resolution_map_ECAL_phi = cms.string('RecoParticleFlow/PFBlockProducer/data/resmap_ECAL_phi.dat'),
                                   
    # max chi2 for element associations in PFBlocks
    pf_chi2_ECAL_Track = cms.double(0.0),
    pf_chi2_HCAL_Track = cms.double(0.0),
    pf_chi2_PS_Track = cms.double(0.0),
    pf_chi2_ECAL_HCAL = cms.double(0.0),
    pf_chi2_PSH_PSV = cms.double(0.0),
    pf_chi2_ECAL_PS = cms.double(0.0),
    pf_chi2_ECAL_GSF = cms.double(0.0),
    pf_multilink = cms.bool(True),

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
    pf_DPtoverPt_Cut = cms.vdouble(1.0,1.0,0.80,0.50,0.00),
    pf_NHit_Cut = cms.vuint32(3,3,3,6,100),
                                   
    # Run particle flow at HLT (hence no RecMuons, no GSF tracks)
    usePFatHLT = cms.bool(False),

    # Debug flag
    debug = cms.untracked.bool(False)
)


