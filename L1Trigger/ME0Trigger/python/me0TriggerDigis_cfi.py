import FWCore.ParameterSet.Config as cms

me0TriggerDigis = cms.EDProducer("ME0TriggerProducer",
    ME0PadDigiProducer = cms.InputTag("simMuonME0PadDigis"),
    tmbParam = cms.PSet()
)
