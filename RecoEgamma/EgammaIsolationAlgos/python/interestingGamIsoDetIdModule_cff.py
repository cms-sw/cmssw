import FWCore.ParameterSet.Config as cms


interestingGamIsoDetId = cms.EDProducer("GamIsoDetIdCollectionProducer",
    recHitsLabel = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    emObjectLabel = cms.InputTag("photons"),
    etCandCut = cms.double(0.0),
    energyCut = cms.double(0.040),
    etCut = cms.double(0),
    outerRadius = cms.double(0.6),
    innerRadius = cms.double(0.0),
    interestingDetIdCollection = cms.string(''),

    severityLevelCut = cms.int32(4),
#     severityRecHitThreshold = cms.double(5.0),
#     spikeIdString = cms.string('kSwissCrossBordersIncluded'),
#     spikeIdThreshold = cms.double(0.95),

    recHitFlagsToBeExcluded = cms.vstring(
       'kFaultyHardware',
       'kPoorCalib',
#        ecalRecHitFlag_kSaturated,
#        ecalRecHitFlag_kLeadingEdgeRecovered,
#        ecalRecHitFlag_kNeighboursRecovered,
       'kTowerRecovered',
       'kDead'
    ),
)
