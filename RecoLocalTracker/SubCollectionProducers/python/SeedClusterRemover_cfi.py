import FWCore.ParameterSet.Config as cms

seedClusterRemover = cms.EDProducer("SeedClusterRemover",
                                    trajectories = cms.InputTag("initialStepSeeds"),
                                    stripClusters = cms.InputTag("siStripClusters"),
                                    pixelClusters = cms.InputTag("siPixelClusters"),
                                    Common = cms.PSet(    maxChi2 = cms.double(9.0)    ), #dummy
                                    )

