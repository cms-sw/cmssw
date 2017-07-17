import FWCore.ParameterSet.Config as cms

me0TriggerDigis = cms.EDProducer("ME0TriggerProducer",
    ME0PadDigiClusterProducer = cms.InputTag("simMuonME0PadDigiClusters"),
    tmbParam = cms.PSet()
)
