import FWCore.ParameterSet.Config as cms

logErrorTooManyClusters = cms.EDFilter("LogErrorEventFilter",
                                       src = cms.InputTag("logErrorHarvester"),
                                       maxErrorFractionInLumi = cms.double(1.0),
                                       maxErrorFractionInRun  = cms.double(1.0),
                                       maxSavedEventsPerLumiAndError = cms.uint32(100000),
                                       categoriesToWatch = cms.vstring("TooManyClusters"),
                                       modulesToIgnore = cms.vstring("SeedGeneratorFromRegionHitsEDProducer:regionalCosmicTrackerSeeds",
                                                                     "PhotonConversionTrajectorySeedProducerFromSingleLeg:photonConvTrajSeedFromSingleLeg")
                                       )


logErrorTooManyTripletsPairs = cms.EDFilter("LogErrorEventFilter",
                                            src = cms.InputTag("logErrorHarvester"),
                                            maxErrorFractionInLumi = cms.double(1.0),
                                            maxErrorFractionInRun  = cms.double(1.0),
                                            maxSavedEventsPerLumiAndError = cms.uint32(100000),
                                            categoriesToWatch = cms.vstring("TooManyTriplets","TooManyPairs","PixelTripletHLTGenerator"),
                                            modulesToIgnore = cms.vstring("SeedGeneratorFromRegionHitsEDProducer:regionalCosmicTrackerSeeds",
                                                                          "PhotonConversionTrajectorySeedProducerFromSingleLeg:photonConvTrajSeedFromSingleLeg")
                                            )


logErrorTooManySeeds = cms.EDFilter("LogErrorEventFilter",
                                    src = cms.InputTag("logErrorHarvester"),
                                    maxErrorFractionInLumi = cms.double(1.0),
                                    maxErrorFractionInRun  = cms.double(1.0),
                                    maxSavedEventsPerLumiAndError = cms.uint32(100000),
                                    categoriesToWatch = cms.vstring("TooManySeeds"),
                                    modulesToIgnore = cms.vstring("SeedGeneratorFromRegionHitsEDProducer:regionalCosmicTrackerSeeds")
                                    )


logErrorTooManySeedsDefault = cms.EDFilter("LogErrorEventFilter",
                                           src = cms.InputTag("logErrorHarvester"),
                                           maxErrorFractionInLumi = cms.double(1.0),
                                           maxErrorFractionInRun  = cms.double(1.0),
                                           maxSavedEventsPerLumiAndError = cms.uint32(100000),
                                           categoriesToWatch = cms.vstring("TooManySeeds"),
                                           )


manystripclus53X = cms.EDFilter('ByClusterSummaryMultiplicityPairEventFilter',
                                multiplicityConfig = cms.PSet(
                                                     firstMultiplicityConfig = cms.PSet(
                                                     clusterSummaryCollection = cms.InputTag("clusterSummaryProducer"),
                                                     subDetEnum = cms.int32(5),
                                                     varEnum = cms.int32(0)
                                                     ),
                                                     secondMultiplicityConfig = cms.PSet(
                                                     clusterSummaryCollection = cms.InputTag("clusterSummaryProducer"),
                                                     subDetEnum = cms.int32(0),
                                                     varEnum = cms.int32(0)
                                                     ),
                                                     ),
                                                     cut = cms.string("( mult2 > 20000+7*mult1)")
                                                     )


toomanystripclus53X = cms.EDFilter('ByClusterSummaryMultiplicityPairEventFilter',
                                   multiplicityConfig = cms.PSet(
                                                        firstMultiplicityConfig = cms.PSet(
                                                        clusterSummaryCollection = cms.InputTag("clusterSummaryProducer"),
                                                        subDetEnum = cms.int32(5),
                                                        varEnum = cms.int32(0)
                                                        ),
                                                        secondMultiplicityConfig = cms.PSet(
                                                        clusterSummaryCollection = cms.InputTag("clusterSummaryProducer"),
                                                        subDetEnum = cms.int32(0),
                                                        varEnum = cms.int32(0)
                                                        ),
                                                        ),
                                                        cut = cms.string("(mult2>50000) && ( mult2 > 20000+7*mult1)")
                                                        )



logErrorTooManyTripletsPairsMainIterations = cms.EDFilter("LogErrorEventFilter",
                                                          src = cms.InputTag("logErrorHarvester"),
                                                          maxErrorFractionInLumi = cms.double(1.0),
                                                          maxErrorFractionInRun  = cms.double(1.0),
                                                          maxSavedEventsPerLumiAndError = cms.uint32(100000),
                                                          categoriesToWatch = cms.vstring("TooManyTriplets","TooManyPairs","PixelTripletHLTGenerator"),
                                                          modulesToWatch = cms.vstring("SeedGeneratorFromRegionHitsEDProducer:initialStepSeeds",
                                                                                       "SeedGeneratorFromRegionHitsEDProducer:pixelPairStepSeeds"
                                                                                       )
                                                          
                                                          )



logErrorTooManySeedsMainIterations = cms.EDFilter("LogErrorEventFilter",
                                                  src = cms.InputTag("logErrorHarvester"),
                                                  maxErrorFractionInLumi = cms.double(1.0),
                                                  maxErrorFractionInRun  = cms.double(1.0),
                                                  maxSavedEventsPerLumiAndError = cms.uint32(100000),
                                                  categoriesToWatch = cms.vstring("TooManySeeds"),
                                                  modulesToWatch = cms.vstring("CkfTrackCandidateMaker:initialStepTrackCandidate",
                                                                               "CkfTrackCandidateMaker:pixelPairTrackCandidate"
                                                                               )
                                                  )



tobtecfakesfilter = cms.EDFilter("TobTecFakesFilter",
                                 minEta = cms.double(0.9), # beginning of transition region for "jet" search
                                 maxEta = cms.double(1.6), # end of transition region for "jet" search
                                 phiWindow = cms.double(0.7), # size of phi region for "jet" search
                                 filter = cms.bool(True), # if true, only events passing filter (bad events) will pass
                                 trackCollection = cms.InputTag("generalTracks"), # track collection to use
                                 ratioAllCut = cms.double(-1.0), # minimum ratio of TOBTEC-seeded tracks / pixelseeded tracks
                                 ratioJetCut = cms.double(3.0),  # minimum ratio of TOBTEC-seeded tracks / pixelseeded tracks in jet
                                 absJetCut = cms.double(20.0)    # minimum number of TOBTEC-seeded tracks in "
                                 )
