import FWCore.ParameterSet.Config as cms

me0TriggerDigis = cms.EDProducer("ME0TriggerProducer",
    ME0PadDigis = cms.InputTag("simMuonME0PadDigis"),
    tmbParam = cms.PSet(
        verbosity = cms.int32(0)
    )
)
