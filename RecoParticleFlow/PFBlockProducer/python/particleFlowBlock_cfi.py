import FWCore.ParameterSet.Config as cms

particleFlowBlock = cms.EDProducer("PFBlockProducer",
    PFClustersPS = cms.InputTag("particleFlowClusterPS"),
    pf_resolution_map_HCAL_eta = cms.string('RecoParticleFlow/PFBlockProducer/data/resmap_HCAL_eta.dat'),
    pf_chi2_ECAL_PS = cms.double(100.0),
    pf_chi2_PS_Track = cms.double(100.0),
    # Track Quality Cut: Tracks are kept if DPt/Pt < Cut
    pf_DPtoverPt_Cut = cms.double(999.9),
    PFClustersHCAL = cms.InputTag("particleFlowClusterHCAL"),
    RecMuons = cms.InputTag("muons"),
    # input collections ----------------------------------------
    # module label to find input rec tracks
    RecTracks = cms.InputTag("elecpreid"),
    # double  pf_chi2_HCAL_PS    = 0
    pf_chi2_ECAL_Track = cms.double(100.0),
    # particle flow parameters ---------------------------------
    # reconstruction method, see PFAlgo/src/PFBlock.cc 
    # int32   pf_recon_method = 3
    # resolution maps
    pf_resolution_map_ECAL_eta = cms.string('RecoParticleFlow/PFBlockProducer/data/resmap_ECAL_eta.dat'),
    PFNuclear = cms.InputTag("pfNuclear"),
    useNuclear = cms.untracked.bool(False),
    pf_resolution_map_HCAL_phi = cms.string('RecoParticleFlow/PFBlockProducer/data/resmap_HCAL_phi.dat'),
    pf_chi2_HCAL_Track = cms.double(100.0),
    debug = cms.untracked.bool(False),
    # verbosity 
    verbose = cms.untracked.bool(False),
    # module label to find input PFClusters
    PFClustersECAL = cms.InputTag("particleFlowClusterECAL"),
    # if true, using special algorithm to process
    # multiple track associations to the same hcal cluster    
    pf_multilink = cms.bool(True),
    pf_resolution_map_ECAL_phi = cms.string('RecoParticleFlow/PFBlockProducer/data/resmap_ECAL_phi.dat'),
    # max chi2 for element associations in PFBlocks
    pf_chi2_ECAL_HCAL = cms.double(10.0),
    pf_chi2_PSH_PSV = cms.double(5.0)
)


