import FWCore.ParameterSet.Config as cms

allLayer0Taus = cms.EDFilter("PATPFTauCleaner",
    markItems = cms.bool(True),
    saveAll = cms.string(''),
    saveRejected = cms.string(''),
    bitsToIgnore = cms.vstring(),
    wantSummary = cms.untracked.bool(True),
    tauSource = cms.InputTag("pfRecoTauProducer"),
    tauDiscriminatorSource = cms.InputTag("pfRecoTauDiscriminationByIsolation"),
    removeOverlaps = cms.PSet(

    )
)


