import FWCore.ParameterSet.Config as cms
gmtSAMuons = cms.EDProducer('Phase2L1TGMTSAMuonGhostCleaner',
                             barrelPrompt      = cms.InputTag('gmtKMTFMuons:prompt'),
                             barrelDisp        = cms.InputTag('gmtKMTFMuons:displaced'),
                             forward     = cms.InputTag('gmtFwdMuons:prompt')
)

