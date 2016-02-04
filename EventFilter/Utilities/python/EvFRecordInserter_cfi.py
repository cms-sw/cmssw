import FWCore.ParameterSet.Config as cms

EvFRecordInserter = cms.EDAnalyzer( "EvFRecordInserter",
                                    inputTag = cms.InputTag("source")
)

