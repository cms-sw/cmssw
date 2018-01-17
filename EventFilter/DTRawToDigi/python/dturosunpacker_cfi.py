import FWCore.ParameterSet.Config as cms

dturosunpacker = cms.EDProducer("DTuROSRawToDigi",
                                  inputLabel = cms.InputTag("source"),
                                  debug = cms.untracked.bool(False),
                               )
