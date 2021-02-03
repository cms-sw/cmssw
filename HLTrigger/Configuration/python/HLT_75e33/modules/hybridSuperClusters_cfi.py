import FWCore.ParameterSet.Config as cms

hybridSuperClusters = cms.EDProducer("UnifiedSCCollectionProducer",
    bcCollection = cms.string('hybridBarrelBasicClusters'),
    bcCollectionUncleanOnly = cms.string('uncleanOnlyHybridBarrelBasicClusters'),
    cleanBcCollection = cms.InputTag("cleanedHybridSuperClusters","hybridBarrelBasicClusters"),
    cleanScCollection = cms.InputTag("cleanedHybridSuperClusters"),
    scCollection = cms.string(''),
    scCollectionUncleanOnly = cms.string('uncleanOnlyHybridSuperClusters'),
    uncleanBcCollection = cms.InputTag("uncleanedHybridSuperClusters","hybridBarrelBasicClusters"),
    uncleanScCollection = cms.InputTag("uncleanedHybridSuperClusters")
)
