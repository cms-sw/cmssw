import FWCore.ParameterSet.Config as cms

particleFlowBlock = cms.EDProducer("PFBlockProducer",
    pf_resolution_map_HCAL_eta = cms.string('RecoParticleFlow/PFBlockProducer/data/resmap_HCAL_eta.dat'),
    pf_chi2_PS_Track = cms.double(100.0),
    # verbosity 
    verbose = cms.untracked.bool(False),
    RecMuons = cms.InputTag("muons"),
    PFConversions = cms.InputTag("pfConversions"),
    useConversions = cms.bool(False),
    pf_resolution_map_HCAL_phi = cms.string('RecoParticleFlow/PFBlockProducer/data/resmap_HCAL_phi.dat'),
    # module label to find input PFClusters
    PFClustersECAL = cms.InputTag("particleFlowClusterECAL"),
    # if true, using special algorithm to process
    # multiple track associations to the same hcal cluster    
    pf_multilink = cms.bool(True),
    # max chi2 for element associations in PFBlocks
    pf_chi2_ECAL_HCAL = cms.double(10.0),
    PFClustersHCAL = cms.InputTag("particleFlowClusterHCAL"),
    PFClustersPS = cms.InputTag("particleFlowClusterPS"),
    pf_chi2_HCAL_Track = cms.double(100.0),
    useNuclear = cms.bool(False),
    # double  pf_chi2_HCAL_PS    = 0
    pf_chi2_ECAL_Track = cms.double(100.0),
    pf_chi2_PSH_PSV = cms.double(5.0),
    pf_chi2_ECAL_PS = cms.double(100.0),
    # Track Quality Cut: Tracks are kept if DPt/Pt < Cut
    pf_DPtoverPt_Cut = cms.double(999.9),
    GsfRecTracks = cms.InputTag("pfTrackElec"),
    # input collections ----------------------------------------
    # module label to find input rec tracks
    RecTracks = cms.InputTag("elecpreid"),
    # particle flow parameters ---------------------------------
    # reconstruction method, see PFAlgo/src/PFBlock.cc 
    # int32   pf_recon_method = 3
    # resolution maps
    pf_resolution_map_ECAL_eta = cms.string('RecoParticleFlow/PFBlockProducer/data/resmap_ECAL_eta.dat'),
    PFNuclear = cms.InputTag("pfNuclear"),
    pf_resolution_map_ECAL_phi = cms.string('RecoParticleFlow/PFBlockProducer/data/resmap_ECAL_phi.dat'),
    debug = cms.untracked.bool(False)
)


