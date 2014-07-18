import FWCore.ParameterSet.Config as cms

EvFFEDSelector = cms.EDProducer( "EvFFEDSelector",
                                 fedList = cms.vuint32(812,1023),
                                 inputTag = cms.InputTag("source")
)

