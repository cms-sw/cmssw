import FWCore.ParameterSet.Config as cms

pfClusterShapes = cms.EDProducer("PFClusterShapeProducer",
    PFClusterShapesLabel = cms.string('ECAL'),
    PFClustersECAL = cms.InputTag("particleFlowCluster","ECAL"),
    PFRecHitsECAL = cms.InputTag("particleFlowCluster","ECAL"),
    useFractions = cms.bool(False),
    W0 = cms.double(4.2)
)


