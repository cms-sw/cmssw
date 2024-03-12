import FWCore.ParameterSet.Config as cms

siPixelRecHits = cms.EDProducer("SiPixelRecHitConverter",
    CPE = cms.string('PixelCPEGeneric'),
    VerboseLevel = cms.untracked.int32(0),
    src = cms.InputTag("siPixelClusters")
)
# foo bar baz
# kJg5BaMLWbg8y
# t4O4Cm3sUtEz3
