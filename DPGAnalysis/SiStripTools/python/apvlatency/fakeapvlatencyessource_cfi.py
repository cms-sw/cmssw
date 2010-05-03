import FWCore.ParameterSet.Config as cms

fakeapvlatency =cms.ESProducer("FakeAPVLatencyESSource",
                               APVLatency = cms.untracked.int32(147)
                               )

