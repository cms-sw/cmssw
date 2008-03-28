import FWCore.ParameterSet.Config as cms

genEventPdfInfo = cms.EDProducer("GenEventPdfInfoProducer",
    src = cms.InputTag("source")
)


