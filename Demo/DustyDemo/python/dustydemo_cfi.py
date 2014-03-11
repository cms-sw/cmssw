import FWCore.ParameterSet.Config as cms

demo = cms.EDAnalyzer('DustyDemo',
	muonSource = cms.InputTag("l1extraParticles"),
	nonIsolatedEmSource = cms.InputTag("l1extraParticles","NonIsolated"),
	etMissSource = cms.InputTag("l1extraParticles","MET"),
	htMissSource = cms.InputTag("l1extraParticles","MHT"),
	forwardJetSource = cms.InputTag("l1extraParticles","Forward"),
	centralJetSource = cms.InputTag("l1extraParticles","Central"),
	tauJetSource = cms.InputTag("l1extraParticles","Tau"),
	hfRingsSource = cms.InputTag("l1extraParticles"),
	particleMapSource = cms.InputTag("l1extraParticleMap"),
	isolatedEmSource = cms.InputTag("l1extraParticles","Isolated")
)
