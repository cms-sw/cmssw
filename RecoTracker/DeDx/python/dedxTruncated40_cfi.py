import FWCore.ParameterSet.Config as cms

dedxTruncated40 = cms.EDProducer("DeDxEstimatorProducer",
    trackDeDxHits = cms.InputTag("dedxHits"),
    fraction = cms.double(0.4),
    estimator = cms.string('truncated')
)


