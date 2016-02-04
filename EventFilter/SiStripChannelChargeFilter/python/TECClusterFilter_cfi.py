import FWCore.ParameterSet.Config as cms

# Cluster Filter
TECClusterFilter = cms.EDFilter("TECClusterFilter",
    # detector modules to be excluded
    ModulesToBeExcluded = cms.vuint32(),
    ClusterProducer = cms.string('siStripClusters325'),
    # with clusters over above respective threshold
    MinNrOfTECClusters = cms.int32(1),
    # thresholds
    ChargeThresholdTEC = cms.int32(20)
)


