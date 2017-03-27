import FWCore.ParameterSet.Config as cms

particleFlowClusterECALMatchedToPhotons = cms.EDProducer('PFClusterMatchedToPhotonsSelector',
                                                         pfClustersTag = cms.InputTag('particleFlowClusterECAL'),
                                                         trackingParticleTag = cms.InputTag('mix', 'MergedTrackTruth'),
                                                         maxDR = cms.double(0.3),
                                                         )



