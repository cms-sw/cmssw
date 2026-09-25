import FWCore.ParameterSet.Config as cms

hltMtdClusters =  cms.EDProducer('MTDClusterProducer',
                                 srcBarrel = cms.InputTag('hltMtdRecHits', 'FTLBarrel'),
                                 srcEndcap = cms.InputTag('hltMtdRecHits', 'FTLEndcap'),
                                 BarrelClusterName = cms.string('FTLBarrel'),
                                 EndcapClusterName = cms.string('FTLEndcap'),
                                 ClusterMode = cms.string('MTDThresholdClusterizer'),
                                 HitThreshold = cms.double(0),
                                 SeedThreshold = cms.double(0),
                                 ClusterThreshold = cms.double(0),
                                 TimeThreshold = cms.double(10),
                                 PositionThreshold = cms.double(-1)
                                 )
