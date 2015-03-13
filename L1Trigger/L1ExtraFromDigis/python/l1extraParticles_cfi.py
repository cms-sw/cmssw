import FWCore.ParameterSet.Config as cms

l1extraParticles = cms.EDProducer("L1ExtraParticlesProd",
    muonSource = cms.InputTag("gtDigis"),
    etTotalSource = cms.InputTag("gctDigis"),
    nonIsolatedEmSource = cms.InputTag("gctDigis","nonIsoEm"),
    etMissSource = cms.InputTag("gctDigis"),
    htMissSource = cms.InputTag("gctDigis"),
    produceMuonParticles = cms.bool(True),
    forwardJetSource = cms.InputTag("gctDigis","forJets"),
    centralJetSource = cms.InputTag("gctDigis","cenJets"),
    produceCaloParticles = cms.bool(True),
    tauJetSource = cms.InputTag("gctDigis","tauJets"),
    isoTauJetSource = cms.InputTag("gctDigis","isoTauJets"),
    isolatedEmSource = cms.InputTag("gctDigis","isoEm"),
    etHadSource = cms.InputTag("gctDigis"),
    hfRingEtSumsSource = cms.InputTag("gctDigis"),
    hfRingBitCountsSource = cms.InputTag("gctDigis"),
    centralBxOnly = cms.bool(True),
    ignoreHtMiss = cms.bool(False)
)


