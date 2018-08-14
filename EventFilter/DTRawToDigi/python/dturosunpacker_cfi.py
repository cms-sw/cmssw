import FWCore.ParameterSet.Config as cms

dturosunpacker = cms.EDProducer("DTuROSRawToDigi",
                                  inputLabel = cms.InputTag("rawDataCollector"),
                                  debug = cms.untracked.bool(False),
                               )
