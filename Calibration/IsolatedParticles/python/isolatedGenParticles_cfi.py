import FWCore.ParameterSet.Config as cms

isolatedGenParticles = cms.EDAnalyzer("IsolatedGenParticles",
                                      GenSrc                = cms.untracked.string("genParticles"),
                                      UseHepMC              = cms.untracked.bool(False),
                                      ChargedHadronSeedP    = cms.untracked.double(1.0),
                                      PTMin                 = cms.untracked.double(1.0),
                                      MaxChargedHadronEta   = cms.untracked.double(2.5),
                                      ConeRadius            = cms.untracked.double(34.98),
                                      ConeRadiusMIP         = cms.untracked.double(14.0),
                                      UseConeIsolation      = cms.untracked.bool(True),
                                      PMaxIsolation         = cms.untracked.double(20.0),
                                      Verbosity             = cms.untracked.int32(0),
                                      DebugL1Info           = cms.untracked.bool(False),
                                      L1extraTauJetSource   = cms.InputTag("l1extraParticles", "Tau"),
                                      L1extraCenJetSource   = cms.InputTag("l1extraParticles", "Central"),
                                      L1extraFwdJetSource   = cms.InputTag("l1extraParticles", "Forward"),
                                      L1extraMuonSource     = cms.InputTag("l1extraParticles"),
                                      L1extraIsoEmSource    = cms.InputTag("l1extraParticles","Isolated"),
                                      L1extraNonIsoEmSource = cms.InputTag("l1extraParticles","NonIsolated"),
                                      L1GTReadoutRcdSource  = cms.InputTag("gtDigis"),
                                      L1GTObjectMapRcdSource= cms.InputTag("hltL1GtObjectMap")
)
