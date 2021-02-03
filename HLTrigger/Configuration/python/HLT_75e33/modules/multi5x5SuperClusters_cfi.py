import FWCore.ParameterSet.Config as cms

multi5x5SuperClusters = cms.EDProducer("UnifiedSCCollectionProducer",
    bcCollection = cms.string('multi5x5EndcapBasicClusters'),
    bcCollectionUncleanOnly = cms.string('uncleanOnlyMulti5x5EndcapBasicClusters'),
    cleanBcCollection = cms.InputTag("multi5x5BasicClustersCleaned","multi5x5EndcapBasicClusters"),
    cleanScCollection = cms.InputTag("multi5x5SuperClustersCleaned","multi5x5EndcapSuperClusters"),
    scCollection = cms.string('multi5x5EndcapSuperClusters'),
    scCollectionUncleanOnly = cms.string('uncleanOnlyMulti5x5EndcapSuperClusters'),
    uncleanBcCollection = cms.InputTag("multi5x5BasicClustersUncleaned","multi5x5EndcapBasicClusters"),
    uncleanScCollection = cms.InputTag("multi5x5SuperClustersUncleaned","multi5x5EndcapSuperClusters")
)
