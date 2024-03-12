import FWCore.ParameterSet.Config as cms

hltElePixelSeedsCombinedUnseeded = cms.EDProducer("SeedCombiner",
    seedCollections = cms.VInputTag("hltElePixelSeedsDoubletsUnseeded", "hltElePixelSeedsTripletsUnseeded")
)
# foo bar baz
# D3VE9PidikRzW
