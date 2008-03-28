import FWCore.ParameterSet.Config as cms

genEventRunInfo = cms.EDProducer("GenEventRunInfoProducer",
    src = cms.InputTag("source")
)


