import FWCore.ParameterSet.Config as cms
from CondCore.DBCommon.CondDBCommon_cfi import CondDBCommon
import os

def find_in_path(name, paths):
    for path in paths.split(':'):
        for root, dirs, files in os.walk(path):
            if name in files:
                return os.path.join(root, name)
            
the_path = [os.environ['CMSSW_BASE'],os.environ['CMSSW_RELEASE_BASE']]
the_path = ':'.join([ '%s/src/RecoEcal/EgammaClusterProducers/data/'%a for a in the_path])
LocalPFECALGBRESSource = cms.ESSource(
    "PoolDBESSource",
    CondDBCommon,
    DumpStat=cms.untracked.bool(False),
    toGet = cms.VPSet(
    cms.PSet(
    record = cms.string('GBRWrapperRcd'),
    tag = cms.string('pfecalsc_EBCorrection'),
    label = cms.untracked.string('pfecalsc_EBCorrection')
    ),        
    cms.PSet(
    record = cms.string('GBRWrapperRcd'),
    tag = cms.string('pfecalsc_EECorrection'),
        label = cms.untracked.string('pfecalsc_EECorrection')
      ),     
    )
)

GBRPrefer = cms.ESPrefer(
    'PoolDBESSource',
    'GlobalTag',
    GBRWrapperRcd = cms.vstring('GBRForest/pfecalsc_EBCorrection',
                                'GBRForest/pfecalsc_EECorrection')
)

to_connect = find_in_path('pfecalsc_regression_weights_01092013.db',the_path)
if to_connect is not None:
    LocalPFECALGBRESSource.connect = 'sqlite_file:%s'%to_connect
else:
    del LocalPFECALGBRESSource
    del GBRPrefer

particleFlowSuperClusterECALBox = cms.EDProducer(
    "PFECALSuperClusterProducer",
    # verbosity 
    verbose = cms.untracked.bool(False),
    #clustering type: "Box" or "Mustache"
    ClusteringType = cms.string("Box"),
    #energy weighting: "Raw", "CalibratedNoPS", "CalibratedTotal"
    EnergyWeight = cms.string("Raw"),

    #this overrides both dphi cuts below if true!
    useDynamicDPhiWindow = cms.bool(False),
    
    #PFClusters collection
    PFClusters = cms.InputTag("particleFlowClusterECAL"),
    PFClustersES = cms.InputTag("particleFlowClusterPS"),
                                              
    PFBasicClusterCollectionBarrel = cms.string("particleFlowBasicClusterECALBarrel"),                                       
    PFSuperClusterCollectionBarrel = cms.string("particleFlowSuperClusterECALBarrel"),
    PFBasicClusterCollectionEndcap = cms.string("particleFlowBasicClusterECALEndcap"),                                       
    PFSuperClusterCollectionEndcap = cms.string("particleFlowSuperClusterECALEndcap"),
    PFBasicClusterCollectionPreshower = cms.string("particleFlowSuperClusterECALPreshower"),
    PFSuperClusterCollectionEndcapWithPreshower = cms.string("particleFlowSuperClusterECALEndcapWithPreshower"),                                          

    #use preshower ?
    use_preshower = cms.bool(True),

    # are the seed thresholds Et or Energy?
    seedThresholdIsET = cms.bool(True),

    # regression setup
    useRegression = cms.bool(False), #regressions are mustache only
    regressionKeyEB = cms.string('pfecalsc_EBCorrection'),
    regressionKeyEE = cms.string('pfecalsc_EECorrection'),
    
    # threshold in ECAL
    thresh_PFClusterSeedBarrel = cms.double(3.0),
    thresh_PFClusterBarrel = cms.double(0.5),

    thresh_PFClusterSeedEndcap = cms.double(5.0),
    thresh_PFClusterEndcap = cms.double(0.5),

    # window width in ECAL
    phiwidth_SuperClusterBarrel = cms.double(0.28),
    etawidth_SuperClusterBarrel = cms.double(0.04),

    phiwidth_SuperClusterEndcap = cms.double(0.28),
    etawidth_SuperClusterEndcap = cms.double(0.04),

    # threshold in preshower
    thresh_PFClusterES = cms.double(0.),                                          

    # turn on merging of the seed cluster to its nearest neighbors
    # that share a rechit
    doSatelliteClusterMerge = cms.bool(False),
    satelliteClusterSeedThreshold = cms.double(50.0),
    satelliteMajorityFraction = cms.double(0.5),
    #thresh_PFClusterMustacheOutBarrel = cms.double(0.),
    #thresh_PFClusterMustacheOutEndcap = cms.double(0.),                                             

    #corrections
    applyCrackCorrections = cms.bool(False)                                          
                                              
)

particleFlowSuperClusterECALMustache = cms.EDProducer(
    "PFECALSuperClusterProducer",
    # verbosity 
    verbose = cms.untracked.bool(False),
    #clustering type: "Box" or "Mustache"
    ClusteringType = cms.string("Mustache"),
    #energy weighting: "Raw", "CalibratedNoPS", "CalibratedTotal"
    EnergyWeight = cms.string("Raw"),

    #this overrides both dphi cuts below if true!
    useDynamicDPhiWindow = cms.bool(True), 
                                              
    #PFClusters collection
    PFClusters = cms.InputTag("particleFlowClusterECAL"),
    PFClustersES = cms.InputTag("particleFlowClusterPS"),
                                              
    PFBasicClusterCollectionBarrel = cms.string("particleFlowBasicClusterECALBarrel"),                                       
    PFSuperClusterCollectionBarrel = cms.string("particleFlowSuperClusterECALBarrel"),
    PFBasicClusterCollectionEndcap = cms.string("particleFlowBasicClusterECALEndcap"),                                       
    PFSuperClusterCollectionEndcap = cms.string("particleFlowSuperClusterECALEndcap"),
    PFBasicClusterCollectionPreshower = cms.string("particleFlowSuperClusterECALPreshower"),
    PFSuperClusterCollectionEndcapWithPreshower = cms.string("particleFlowSuperClusterECALEndcapWithPreshower"),                                          

    #use preshower ?
    use_preshower = cms.bool(True),

    # are the seed thresholds Et or Energy?
    seedThresholdIsET = cms.bool(True),
    # regression setup
    useRegression = cms.bool(True),
    regressionKeyEB = cms.string('pfecalsc_EBCorrection'),
    regressionKeyEE = cms.string('pfecalsc_EECorrection'),
    
    # threshold in ECAL
    thresh_PFClusterSeedBarrel = cms.double(3.0),
    thresh_PFClusterBarrel = cms.double(0.0),

    thresh_PFClusterSeedEndcap = cms.double(5.0),
    thresh_PFClusterEndcap = cms.double(0.0),

    # window width in ECAL ( these don't mean anything for Mustache )
    phiwidth_SuperClusterBarrel = cms.double(0.6),
    etawidth_SuperClusterBarrel = cms.double(0.04),

    phiwidth_SuperClusterEndcap = cms.double(0.6),
    etawidth_SuperClusterEndcap = cms.double(0.04),

    # threshold in preshower
    thresh_PFClusterES = cms.double(0.),           

    # turn on merging of the seed cluster to its nearest neighbors
    # that share a rechit
    doSatelliteClusterMerge = cms.bool(False),
    satelliteClusterSeedThreshold = cms.double(50.0),
    satelliteMajorityFraction = cms.double(0.5),
    #thresh_PFClusterMustacheOutBarrel = cms.double(0.),
    #thresh_PFClusterMustacheOutEndcap = cms.double(0.), 

    #corrections
    applyCrackCorrections = cms.bool(False)
                                              
)

#define the default clustering type
particleFlowSuperClusterECAL = particleFlowSuperClusterECALMustache.clone()
