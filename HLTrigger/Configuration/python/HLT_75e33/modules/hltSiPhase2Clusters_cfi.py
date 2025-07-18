import FWCore.ParameterSet.Config as cms

# Clusterizer options
hltSiPhase2Clusters = cms.EDProducer('Phase2TrackerClusterizer',
    src = cms.InputTag("mix", "Tracker"),
    maxClusterSize = cms.uint32(0), # was 8
    maxNumberClusters = cms.uint32(0)
)

from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(hltSiPhase2Clusters, src = "mixData:Tracker")
