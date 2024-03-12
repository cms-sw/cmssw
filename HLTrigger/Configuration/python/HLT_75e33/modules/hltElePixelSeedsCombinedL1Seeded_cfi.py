import FWCore.ParameterSet.Config as cms

hltElePixelSeedsCombinedL1Seeded = cms.EDProducer("SeedCombiner",
    seedCollections = cms.VInputTag("hltElePixelSeedsDoubletsL1Seeded", "hltElePixelSeedsTripletsL1Seeded")
)
# foo bar baz
# hgmbapFQXsuyo
# 1C4qUYpQ1Bn98
