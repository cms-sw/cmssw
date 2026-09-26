import FWCore.ParameterSet.Config as cms

TTClustersFromPhase2TrackerDigis = cms.EDProducer("TTClusterBuilder_Phase2TrackerDigi_",
    rawHits = cms.VInputTag(cms.InputTag("mix","Tracker")),
    ADCThreshold = cms.uint32(30)
)

from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(TTClustersFromPhase2TrackerDigis, rawHits = ["mixData:Tracker"])
