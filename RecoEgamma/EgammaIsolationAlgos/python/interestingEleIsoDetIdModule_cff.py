import FWCore.ParameterSet.Config as cms

interestingEleIsoDetId = cms.EDProducer("EleIsoDetIdCollectionProducer",
    recHitsLabel = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    emObjectLabel = cms.InputTag("gsfElectrons"),
    etCandCut = cms.double(0.0),
    energyCut = cms.double(0.040),
    etCut = cms.double(0),
    outerRadius = cms.double(0.6),
    innerRadius = cms.double(0.0),
    interestingDetIdCollection = cms.string(''),

    severityLevelCut = cms.int32(3),
    severityRecHitThreshold = cms.double(5.0),
    spikeIdString = cms.string('kSwissCross'),
    spikeIdThreshold = cms.double(0.95)
)
