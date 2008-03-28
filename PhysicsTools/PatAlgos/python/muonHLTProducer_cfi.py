import FWCore.ParameterSet.Config as cms

HLT1MuonIso = cms.EDProducer("PATHLTProducer",
    triggerResults = cms.InputTag("TriggerResults","","HLT"),
    filterName = cms.InputTag("SingleMuIsoL3IsoFiltered","","HLT"),
    triggerName = cms.string('HLT1MuonIso')
)

HLT1MuonNonIso = cms.EDProducer("PATHLTProducer",
    triggerResults = cms.InputTag("TriggerResults","","HLT"),
    filterName = cms.InputTag("SingleMuNoIsoL3PreFiltered","","HLT"),
    triggerName = cms.string('HLT1MuonNonIso')
)

HLT2MuonNonIso = cms.EDProducer("PATHLTProducer",
    triggerResults = cms.InputTag("TriggerResults","","HLT"),
    filterName = cms.InputTag("DiMuonNoIsoL3PreFiltered","","HLT"),
    triggerName = cms.string('HLT2MuonNonIso')
)

muonHLTProducer = cms.Sequence(HLT1MuonIso*HLT1MuonNonIso*HLT2MuonNonIso)

