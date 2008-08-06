import FWCore.ParameterSet.Config as cms

hltResults = cms.EDFilter("HLTJetMETDQMSource",
    plotAll = cms.untracked.bool(False),
    ptMax = cms.untracked.double(100.0),
    ptMin = cms.untracked.double(0.0),
    filters = cms.VPSet(),
    triggerSummaryLabel = cms.InputTag("triggerSummaryProducerAOD","","FU")
)


