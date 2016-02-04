import FWCore.ParameterSet.Config as cms

hltEventInfoClient = cms.EDAnalyzer("HLTEventInfoClient",
    prescaleLS = cms.untracked.int32(-1),
    monitorDir = cms.untracked.string(''),
    prescaleEvt = cms.untracked.int32(1)
)


