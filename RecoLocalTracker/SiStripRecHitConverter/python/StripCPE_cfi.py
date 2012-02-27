import FWCore.ParameterSet.Config as cms

StripCPEESProducer = cms.ESProducer("StripCPEESProducer",
                                    ComponentName = cms.string('SimpleStripCPE'),
                                    ComponentType = cms.string('SimpleStripCPE')
)


