import FWCore.ParameterSet.Config as cms

seedClusterRemoverPhase2 = cms.EDProducer("SeedClusterRemoverPhase2",
                                    trajectories = cms.InputTag("initialStepSeeds"),
                                    phase2OTClusters = cms.InputTag("siPhase2Clusters"),
                                    pixelClusters = cms.InputTag("siPixelClusters"),
                                    )

