import FWCore.ParameterSet.Config as cms

gctCandsFromGt = cms.EDProducer("GtToGctCands",
    inputLabel = cms.InputTag("gtDigis")
)

l1extraParticles = cms.EDProducer("L1ExtraParticlesProd",
    muonSource = cms.InputTag("gtDigis"),
    etTotalSource = cms.InputTag("gctCandsFromGt"),
    nonIsolatedEmSource = cms.InputTag("gctCandsFromGt","nonIsoEm"),
    etMissSource = cms.InputTag("gctCandsFromGt"),
    htMissSource = cms.InputTag("gctCandsFromGt"),
    produceMuonParticles = cms.bool(True),
    forwardJetSource = cms.InputTag("gctCandsFromGt","forJets"),
    centralJetSource = cms.InputTag("gctCandsFromGt","cenJets"),
    produceCaloParticles = cms.bool(True),
    tauJetSource = cms.InputTag("gctCandsFromGt","tauJets"),
    isolatedEmSource = cms.InputTag("gctCandsFromGt","isoEm"),
    etHadSource = cms.InputTag("gctCandsFromGt"),
    hfRingEtSumsSource = cms.InputTag("gctCandsFromGt"),
    hfRingBitCountsSource = cms.InputTag("gctCandsFromGt"),
    centralBxOnly = cms.bool(True),
    ignoreHtMiss = cms.bool(True)
)
