import FWCore.ParameterSet.Config as cms

genEventScale = cms.EDProducer("GenEventScaleProducer",
    src = cms.InputTag("source")
)


