import FWCore.ParameterSet.Config as cms

pfBlockElementSC = cms.EDProducer("PFBlockElementSuperClusterProducer",
                                  ECALSuperClusters=cms.VInputTag(cms.InputTag('correctedHybridSuperClusters'),
                                                                  cms.InputTag('correctedMulti5x5SuperClustersWithPreshower')),
                                  PFBESuperClusters=cms.string('')
                                  )

