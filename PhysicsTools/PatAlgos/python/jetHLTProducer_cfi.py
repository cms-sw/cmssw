import FWCore.ParameterSet.Config as cms

HLT1Jet = cms.EDProducer("PATHLTProducer",
    triggerResults = cms.InputTag("TriggerResults","","HLT"),
    filterName = cms.InputTag("hlt1jet200","","HLT"),
    triggerName = cms.string('HLT1Jet')
)

HLT2Jet = cms.EDProducer("PATHLTProducer",
    triggerResults = cms.InputTag("TriggerResults","","HLT"),
    filterName = cms.InputTag("hlt2jet150","","HLT"),
    triggerName = cms.string('HLT2Jet')
)

HLT3Jet = cms.EDProducer("PATHLTProducer",
    triggerResults = cms.InputTag("TriggerResults","","HLT"),
    filterName = cms.InputTag("hlt3jet85","","HLT"),
    triggerName = cms.string('HLT3Jet')
)

HLT4Jet = cms.EDProducer("PATHLTProducer",
    triggerResults = cms.InputTag("TriggerResults","","HLT"),
    filterName = cms.InputTag("hlt4jet60","","HLT"),
    triggerName = cms.string('HLT4Jet')
)

jetHLTProducer = cms.Sequence(HLT1Jet*HLT2Jet*HLT3Jet*HLT4Jet)

