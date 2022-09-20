import FWCore.ParameterSet.Config as cms

siPhase2Clusters = cms.EDProducer("Phase2TrackerClusterizer",
    maxClusterSize = cms.uint32(0),
    maxNumberClusters = cms.uint32(0),
    src = cms.InputTag("mix","Tracker")
)

from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(siPhase2Clusters, src = "mixData:Tracker")
