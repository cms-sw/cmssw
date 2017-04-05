import FWCore.ParameterSet.Config as cms

particleFlowClusterECALMatchedToPhotons = cms.EDProducer('PFClusterMatchedToPhotonsSelector',
                                                         pfClustersTag = cms.InputTag('particleFlowClusterECAL'),
                                                         genParticleTag = cms.InputTag('genParticles'),
                                                         trackingParticleTag = cms.InputTag('mix', 'MergedTrackTruth'),
                                                         maxDR = cms.double(0.3),
                                                         volumeRadius_EB = cms.double(123.8),
                                                         volumeZ_EB = cms.double(304.5),
                                                         volumeZ_EE = cms.double(317.0)
                                                         )



