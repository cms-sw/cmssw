import FWCore.ParameterSet.Config as cms

stripCPEESProducer = cms.ESProducer("StripCPEESProducer",
                                    ComponentName = cms.string('SimpleStripCPE'),
                                    ComponentType = cms.string('SimpleStripCPE'),
                                    parameters    = cms.PSet()
)
