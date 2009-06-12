import FWCore.ParameterSet.Config as cms

scalersRawToDigi = cms.EDFilter("ScalersRawToDigi",
                                scalersInputTag = cms.InputTag("source")
                                )


