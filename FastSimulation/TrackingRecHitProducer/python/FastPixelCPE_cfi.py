import FWCore.ParameterSet.Config as cms

FastPixelCPEESProducer = cms.ESProducer("FastPixelCPEESProducer",
                                        ComponentName = cms.string('FastPixelCPE'),
                                        )


