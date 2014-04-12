import FWCore.ParameterSet.Config as cms

hltEventInfoClient = cms.EDAnalyzer("HLTEventInfoClient",
    monitorDir = cms.untracked.string(''),
    prescaleLS = cms.untracked.int32(-1),
    prescaleEvt = cms.untracked.int32(1)
)
