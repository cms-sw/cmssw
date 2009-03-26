import FWCore.ParameterSet.Config as cms

l1extratest = cms.EDAnalyzer("L1ExtraTestAnalyzer",
    gtReadoutSource = cms.InputTag("gtDigis"),
    muonSource = cms.InputTag("l1extraParticles"),
    nonIsolatedEmSource = cms.InputTag("l1extraParticles","NonIsolated"),
    etMissSource = cms.InputTag("l1extraParticles","MET"),
    htMissSource = cms.InputTag("l1extraParticles","MHT"),
    forwardJetSource = cms.InputTag("l1extraParticles","Forward"),
    centralJetSource = cms.InputTag("l1extraParticles","Central"),
    tauJetSource = cms.InputTag("l1extraParticles","Tau"),
    particleMapSource = cms.InputTag("l1extraParticleMap"),
    isolatedEmSource = cms.InputTag("l1extraParticles","Isolated")
)


