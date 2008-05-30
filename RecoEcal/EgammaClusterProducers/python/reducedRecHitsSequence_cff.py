import FWCore.ParameterSet.Config as cms

interestingEcalDetIdEB = cms.EDProducer("InterestingDetIdCollectionProducer",
    basicClustersLabel = cms.InputTag("hybridSuperClusters","hybridBarrelBasicClusters"),
    recHitsLabel = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    etaSize = cms.int32(5),
    interestingDetIdCollection = cms.string(''),
    phiSize = cms.int32(5)
)

interestingEcalDetIdEE = cms.EDProducer("InterestingDetIdCollectionProducer",
    basicClustersLabel = cms.InputTag("multi5x5BasicClusters","multi5x5EndcapBasicClusters"),
    recHitsLabel = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    etaSize = cms.int32(5),
    interestingDetIdCollection = cms.string(''),
    phiSize = cms.int32(5)
)

reducedEcalRecHitsEB = cms.EDProducer("ReducedRecHitCollectionProducer",
    recHitsLabel = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    interestingDetIdCollections = cms.VInputTag(cms.InputTag("interestingEcalDetIdEB")),
    reducedHitsCollection = cms.string('')
)

reducedEcalRecHitsEE = cms.EDProducer("ReducedRecHitCollectionProducer",
    recHitsLabel = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    interestingDetIdCollections = cms.VInputTag(cms.InputTag("interestingEcalDetIdEE")),
    reducedHitsCollection = cms.string('')
)

reducedRecHitsSequence = cms.Sequence(interestingEcalDetIdEB*interestingEcalDetIdEE*reducedEcalRecHitsEB*reducedEcalRecHitsEE)

