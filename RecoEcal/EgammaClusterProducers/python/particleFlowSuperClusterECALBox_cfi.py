import FWCore.ParameterSet.Config as cms
import RecoEcal.EgammaClusterProducers.particleFlowSuperClusterECALMustache_cfi as _mod

particleFlowSuperClusterECALBox = _mod.particleFlowSuperClusterECALMustache.clone(
    # verbosity
    verbose = False,
    # clustering type: "Box" or "Mustache"
    ClusteringType = "Box",
    # energy weighting: "Raw", "CalibratedNoPS", "CalibratedTotal"
    EnergyWeight = "Raw",

    # this overrides both dphi cuts below if true!
    useDynamicDPhiWindow = False,

    # PFClusters collection
    PFClusters = "particleFlowClusterECAL",
    ESAssociation = "particleFlowClusterECAL",
    BeamSpot = "offlineBeamSpot",

    PFBasicClusterCollectionBarrel = "particleFlowBasicClusterECALBarrel",
    PFSuperClusterCollectionBarrel = "particleFlowSuperClusterECALBarrel",
    PFBasicClusterCollectionEndcap = "particleFlowBasicClusterECALEndcap",
    PFSuperClusterCollectionEndcap = "particleFlowSuperClusterECALEndcap",
    PFBasicClusterCollectionPreshower = "particleFlowBasicClusterECALPreshower",
    PFSuperClusterCollectionEndcapWithPreshower = "particleFlowSuperClusterECALEndcapWithPreshower",

    # are the seed thresholds Et or Energy?
    seedThresholdIsET = True,

    # regression setup
    useRegression = False, # regressions are mustache only
    regressionConfig = dict(),

    # threshold for final SuperCluster Et
    thresh_SCEt = 4.0,

    # threshold in ECAL
    thresh_PFClusterSeedBarrel = 3.0,
    thresh_PFClusterBarrel = 0.5,

    thresh_PFClusterSeedEndcap = 5.0,
    thresh_PFClusterEndcap = 0.5,

    # window width in ECAL
    phiwidth_SuperClusterBarrel = 0.28,
    etawidth_SuperClusterBarrel = 0.04,

    phiwidth_SuperClusterEndcap = 0.28,
    etawidth_SuperClusterEndcap = 0.04,

    # threshold in preshower
    thresh_PFClusterES = 0.,

    # turn on merging of the seed cluster to its nearest neighbors
    # that share a rechit
    doSatelliteClusterMerge = False,
    satelliteClusterSeedThreshold = 50.0,
    satelliteMajorityFraction = 0.5,
    dropUnseedable = False,

    # corrections
    applyCrackCorrections = False
)
