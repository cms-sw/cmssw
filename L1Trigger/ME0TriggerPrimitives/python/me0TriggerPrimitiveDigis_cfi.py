import FWCore.ParameterSet.Config as cms

me0TriggerPrimitiveDigis = cms.EDProducer("ME0TriggerPrimitivesProducer",
    ME0PadDigiProducer = cms.InputTag("simMuonME0PadDigis"),
    tmbParam = cms.PSet()
)
