import FWCore.ParameterSet.Config as cms

seedmultiplicitymonitor = cms.EDAnalyzer('SeedMultiplicityAnalyzer',
                                         TTRHBuilder = cms.string('WithTrackAngle'),
                                         seedCollections = cms.VPSet(cms.PSet(src=cms.InputTag("initialStepSeeds")),
                                                                     cms.PSet(src=cms.InputTag("lowPtTripletStepSeeds")),
                                                                     cms.PSet(src=cms.InputTag("pixelPairStepSeeds"),
                                                                              maxValue=cms.untracked.double(500000),nBins=cms.untracked.uint32(2000)),
                                                                     cms.PSet(src=cms.InputTag("detachedTripletStepSeeds")),
                                                                     cms.PSet(src=cms.InputTag("mixedTripletStepSeedsA")),
                                                                     cms.PSet(src=cms.InputTag("mixedTripletStepSeedsB")),
                                                                     cms.PSet(src=cms.InputTag("mixedTripletStepSeeds"),
                                                                              maxValue=cms.untracked.double(200000),nBins=cms.untracked.uint32(2000)),
                                                                     cms.PSet(src=cms.InputTag("pixelLessStepSeeds"),
                                                                              maxValue=cms.untracked.double(200000),nBins=cms.untracked.uint32(2000)),
                                                                     cms.PSet(src=cms.InputTag("tobTecStepSeeds"))
                                                                     ),
                                         multiplicityCorrelations = cms.VPSet()
                                         )
