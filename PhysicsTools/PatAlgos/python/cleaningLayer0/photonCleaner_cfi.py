import FWCore.ParameterSet.Config as cms

allLayer0Photons = cms.EDFilter("PATBasePhotonCleaner",
    saveAll = cms.string(''),
    saveRejected = cms.string(''),
    removeElectrons = cms.string('bySeed'),
    wantSummary = cms.untracked.bool(True),
    isolation = cms.PSet(
        hcal = cms.PSet(
            src = cms.InputTag("patAODPhotonIsolations","egammaPhotonTowersDeposits"),
            deltaR = cms.double(0.3),
            cut = cms.double(3.0)
        ),
        tracker = cms.PSet(
            src = cms.InputTag("patAODPhotonIsolations","egammaPhotonTkDeposits"),
            deltaR = cms.double(0.3),
            skipDefaultVeto = cms.bool(True),
            cut = cms.double(5.0),
            vetos = cms.vstring('Threshold(1.5)')
        ),
        ecal = cms.PSet(
            src = cms.InputTag("patAODPhotonIsolations","egammaPhotonEcalDeposits"),
            deltaR = cms.double(0.3),
            cut = cms.double(5.0)
        )
    ),
    markItems = cms.bool(True),
    electrons = cms.InputTag("allLayer0Electrons"),
    bitsToIgnore = cms.vstring('Isolation/All'),
    removeDuplicates = cms.string('bySeed'),
    photonSource = cms.InputTag("photons")
)


