import FWCore.ParameterSet.Config as cms

hltElePixelSeedsCombinedL1Seeded = cms.EDProducer("SeedCombiner",
    seedCollections = cms.VInputTag("hltElePixelSeedsDoubletsL1Seeded", "hltElePixelSeedsTripletsL1Seeded")
)
