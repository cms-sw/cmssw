import FWCore.ParameterSet.Config as cms

me0TriggerDigis = cms.EDProducer("ME0TriggerProducer",
    ME0PadDigiClusters = cms.InputTag("simMuonME0PadDigiClusters"),
    ME0PadDigis = cms.InputTag("simMuonME0PadDigis"),
    useClusters = cms.bool(False),
    tmbParam = cms.PSet()
)
