import FWCore.ParameterSet.Config as cms

hcalPatternSource = cms.EDProducer("HcalPatternSource",
                                   Bunches = cms.untracked.vint32(5),
                                   Presamples = cms.untracked.int32(4),
                                   Samples = cms.untracked.int32(10),
                                   Patterns = cms.untracked.string("/tmp/example/example_crate_0.xml")
                                   )
