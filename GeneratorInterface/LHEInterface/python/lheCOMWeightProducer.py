import FWCore.ParameterSet.Config as cms

lheCOMWeightProducer = cms.EDProducer("LHECOMWeightProducer",
  lheSrc = cms.InputTag("source"),
  NewECMS = cms.double(7000)
)
