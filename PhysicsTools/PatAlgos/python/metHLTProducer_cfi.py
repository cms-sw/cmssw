import FWCore.ParameterSet.Config as cms

HLT1MET = cms.EDProducer("PATHLTProducer",
    triggerResults = cms.InputTag("TriggerResults","","HLT"),
    filterName = cms.InputTag("hlt1MET65","","HLT"),
    triggerName = cms.string('HLT1MET')
)

metHLTProducer = cms.Sequence(HLT1MET)

