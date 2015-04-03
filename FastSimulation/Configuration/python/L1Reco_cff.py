import FWCore.ParameterSet.Config as cms

# the only thing FastSim runs from L1Reco is l1extraParticles
from L1Trigger.Configuration.L1Reco_cff import l1extraParticles

# some collections have different labels
l1extraParticles.isolatedEmSource    = cms.InputTag("simGctDigis","isoEm")
l1extraParticles.nonIsolatedEmSource = cms.InputTag("simGctDigis","nonIsoEm")

l1extraParticles.centralJetSource = cms.InputTag("simGctDigis","cenJets")
l1extraParticles.tauJetSource     = cms.InputTag("simGctDigis","tauJets")
l1extraParticles.isoTauJetSource  = cms.InputTag("simGctDigis","isoTauJets")
l1extraParticles.forwardJetSource = cms.InputTag("simGctDigis","forJets")

l1extraParticles.muonSource = cms.InputTag('simGmtDigis')

l1extraParticles.etTotalSource = cms.InputTag("simGctDigis")
l1extraParticles.etHadSource   = cms.InputTag("simGctDigis")
l1extraParticles.htMissSource  = cms.InputTag("simGctDigis")
l1extraParticles.etMissSource  = cms.InputTag("simGctDigis")

l1extraParticles.hfRingEtSumsSource    = cms.InputTag("simGctDigis")
l1extraParticles.hfRingBitCountsSource = cms.InputTag("simGctDigis")

# must be set to true when used in HLT, as is the case for FastSim
l1extraParticles.centralBxOnly = True

L1Reco = cms.Sequence(l1extraParticles)
