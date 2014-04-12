import FWCore.ParameterSet.Config as cms

# Preshower cluster producer
multi5x5SuperClustersWithPreshower = cms.EDProducer("PreshowerPhiClusterProducer",

                                                    esStripEnergyCut = cms.double(0.0),
                                                    esPhiClusterDeltaEta = cms.double(0.15),
                                                    esPhiClusterDeltaPhi = cms.double(0.12),
                                                    
                                                    preshClusterCollectionY = cms.string('preshowerYClusters'),
                                                    # building endcap association
                                                    assocSClusterCollection = cms.string(''),
                                                    etThresh = cms.double(0.0),
                                                    # building preshower clusters
                                                    preshRecHitProducer = cms.InputTag("ecalPreshowerRecHit","EcalRecHitsES"),
                                                    preshClusterCollectionX = cms.string('preshowerXClusters'),
                                                    
                                                    endcapSClusterProducer = cms.InputTag("multi5x5SuperClusters","multi5x5EndcapSuperClusters")
                                                    )

uncleanedOnlyMulti5x5SuperClustersWithPreshower = multi5x5SuperClustersWithPreshower.clone()

uncleanedOnlyMulti5x5SuperClustersWithPreshower.endcapSClusterProducer = cms.InputTag("multi5x5SuperClusters","uncleanOnlyMulti5x5EndcapSuperClusters")
