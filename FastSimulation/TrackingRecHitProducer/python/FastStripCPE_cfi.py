import FWCore.ParameterSet.Config as cms

FastStripCPEESProducer = cms.ESProducer("FastStripCPEESProducer",
                                        ComponentName = cms.string('FastStripCPE')
                                        )


