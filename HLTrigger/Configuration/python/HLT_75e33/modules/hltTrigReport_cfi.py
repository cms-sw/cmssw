import FWCore.ParameterSet.Config as cms

hltTrigReport = cms.EDAnalyzer("HLTrigReport",
    HLTriggerResults = cms.InputTag("TriggerResults","","HLTX"),
    ReferencePath = cms.untracked.string('HLTriggerFinalPath'),
    ReferenceRate = cms.untracked.double(100.0),
    reportBy = cms.untracked.string('job'),
    resetBy = cms.untracked.string('never'),
    serviceBy = cms.untracked.string('never')
)
