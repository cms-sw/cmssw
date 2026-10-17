import FWCore.ParameterSet.Config as cms

# This is the new clusterizer, that accurately emulates the FE chips,
# unlike the old TTCluster_cfi.py one.

TTClustersFromPhase2TrackerDigis = cms.EDProducer("TTClusterBuilderNew",
    src = cms.InputTag('mix', 'Tracker'),
    maxClusterWidth = cms.uint32(4),
    # Enable vetoes used by MPA chips for clusters in PS-p sensors.
    enableClusterVetoes = cms.bool(True)
)                                                  

from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(TTClustersFromPhase2TrackerDigis, rawHits = ["mixData:Tracker"])
