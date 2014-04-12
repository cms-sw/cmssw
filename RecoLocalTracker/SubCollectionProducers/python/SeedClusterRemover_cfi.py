import FWCore.ParameterSet.Config as cms

seedClusterRemover = cms.EDProducer("SeedClusterRemover",
                                    trajectories = cms.InputTag("initialStepSeeds"),
                                    stripClusters = cms.InputTag("siStripClusters"),
                                    overrideTrkQuals = cms.InputTag(""), #dummy
                                    pixelClusters = cms.InputTag("siPixelClusters"),
                                    Common = cms.PSet(    maxChi2 = cms.double(9.0)    ), #dummy
                                    TrackQuality = cms.string('highPurity'), #dummy
                                    clusterLessSolution = cms.bool(True)
                                    )

