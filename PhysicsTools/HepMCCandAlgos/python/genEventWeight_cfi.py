import FWCore.ParameterSet.Config as cms

genEventWeight = cms.EDProducer("GenEventWeightProducer",
    src = cms.InputTag("source")
)


