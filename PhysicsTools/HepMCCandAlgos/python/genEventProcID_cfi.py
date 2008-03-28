import FWCore.ParameterSet.Config as cms

genEventProcID = cms.EDProducer("GenEventProcIDProducer",
    src = cms.InputTag("source")
)


