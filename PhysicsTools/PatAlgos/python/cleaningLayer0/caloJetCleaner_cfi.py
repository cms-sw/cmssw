import FWCore.ParameterSet.Config as cms

allLayer0Jets = cms.EDFilter("PATCaloJetCleaner",
    selection = cms.PSet(
        type = cms.string('none')
    ),
    jetSource = cms.InputTag("iterativeCone5CaloJets"),
    saveRejected = cms.string(''),
    wantSummary = cms.untracked.bool(True),
    markItems = cms.bool(True),
    saveAll = cms.string(''),
    bitsToIgnore = cms.vstring(),
    removeOverlaps = cms.PSet(
        jets = cms.PSet(
            deltaR = cms.double(0.3),
            collection = cms.InputTag("allLayer0Electrons")
        ),
        taus = cms.PSet(

        ),
        photons = cms.PSet(

        ),
        user = cms.VPSet()
    )
)


