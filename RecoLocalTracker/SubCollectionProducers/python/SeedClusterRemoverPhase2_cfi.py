import FWCore.ParameterSet.Config as cms

seedClusterRemover = cms.EDProducer("SeedClusterRemoverPhase2",
                                    trajectories = cms.InputTag("initialStepSeeds"),
                                    phase2OTClusters = cms.InputTag("siPhase2Clusters"),
                                    overrideTrkQuals = cms.InputTag(""), #dummy
                                    pixelClusters = cms.InputTag("siPixelClusters"),
                                    Common = cms.PSet(    maxChi2 = cms.double(9.0)    ), #dummy
                                    TrackQuality = cms.string('highPurity'), #dummy
                                    clusterLessSolution = cms.bool(True)
                                    )

