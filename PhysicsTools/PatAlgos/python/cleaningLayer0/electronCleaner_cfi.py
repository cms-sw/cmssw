import FWCore.ParameterSet.Config as cms

allLayer0Electrons = cms.EDFilter("PATElectronCleaner",
    selection = cms.PSet(
        type = cms.string('none')
    ),
    saveRejected = cms.string(''),
    bitsToIgnore = cms.vstring('Isolation/All'),
    wantSummary = cms.untracked.bool(True),
    electronSource = cms.InputTag("pixelMatchGsfElectrons"),
    markItems = cms.bool(True),
    saveAll = cms.string(''),
    isolation = cms.PSet(
        tracker = cms.PSet(
            src = cms.InputTag("patAODElectronIsolations","egammaElectronTkIsolation"),
            cut = cms.double(2.0)
        )
    ),
    removeDuplicates = cms.bool(True)
)


