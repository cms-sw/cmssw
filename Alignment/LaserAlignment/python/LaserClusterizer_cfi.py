import FWCore.ParameterSet.Config as cms

# configuration of the Laser Clusterizer
#
siStripClusters = cms.EDFilter("LaserClusterizer",
    ClusterMode = cms.string('LaserBeamClusterizer'),
    DigiProducersList = cms.VPSet(cms.PSet(
        DigiLabel = cms.string('\0'),
        DigiProducer = cms.string('LaserAlignment')
    )),
    # width of the clusters in sigma's
    ClusterWidth = cms.double(1.0),
    BeamFitProducer = cms.string('LaserAlignment')
)


