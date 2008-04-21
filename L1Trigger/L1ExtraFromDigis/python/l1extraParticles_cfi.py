import FWCore.ParameterSet.Config as cms

l1extraParticles = cms.EDProducer("L1ExtraParticlesProd",
    muonSource = cms.InputTag("gmtDigis"),
    etTotalSource = cms.InputTag("gctDigis"),
    nonIsolatedEmSource = cms.InputTag("gctDigis","nonIsoEm"),
    etMissSource = cms.InputTag("gctDigis"),
    produceMuonParticles = cms.bool(True),
    forwardJetSource = cms.InputTag("gctDigis","forJets"),
    centralJetSource = cms.InputTag("gctDigis","cenJets"),
    produceCaloParticles = cms.bool(True),
    tauJetSource = cms.InputTag("gctDigis","tauJets"),
    isolatedEmSource = cms.InputTag("gctDigis","isoEm"),
    etHadSource = cms.InputTag("gctDigis"),
    centralBxOnly = cms.bool(True)
)


