import FWCore.ParameterSet.Config as cms

allLayer0METs = cms.EDFilter("PATCaloMETCleaner",
    metSource = cms.InputTag("corMetType1Icone5"),
    saveAll = cms.string(''),
    bitsToIgnore = cms.vstring(),
    wantSummary = cms.untracked.bool(True),
    markItems = cms.bool(True),
    saveRejected = cms.string('')
)


