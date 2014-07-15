import FWCore.ParameterSet.Config as cms

seedmultiplicitymonitor = cms.EDAnalyzer('SeedMultiplicityAnalyzer',
                                         TTRHBuilder = cms.string('WithTrackAngle'),
                                         seedCollections = cms.VPSet(cms.PSet(src=cms.InputTag("newSeedFromTriplets")),
                                                                     cms.PSet(src=cms.InputTag("newSeedFromPairs")),
                                                                     cms.PSet(src=cms.InputTag("secTriplets")),
                                                                     cms.PSet(src=cms.InputTag("thTripletsA")),
                                                                     cms.PSet(src=cms.InputTag("thTripletsB")),
                                                                     cms.PSet(src=cms.InputTag("thTriplets")),
                                                                     cms.PSet(src=cms.InputTag("fourthPLSeeds")),
                                                                     cms.PSet(src=cms.InputTag("fifthSeeds"))
                                                                         ),
                                         multiplicityCorrelations = cms.VPSet()
                                         )
