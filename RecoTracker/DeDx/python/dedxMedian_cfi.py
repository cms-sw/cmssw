import FWCore.ParameterSet.Config as cms

dedxMedian = cms.EDProducer("DeDxEstimatorProducer",
    trackDeDxHits = cms.InputTag("dedxHits"),
    estimator = cms.string('median')
)


