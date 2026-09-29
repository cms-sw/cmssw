import FWCore.ParameterSet.Config as cms

gmtFwdMuons = cms.EDProducer(
    'Phase2L1TGMTFwdMuonTranslator',
    stubs      = cms.InputTag('gmtStubs','tps'),
    omtfConstrSAMs = cms.InputTag('simOmtfPhase2Digis', 'constrCands'),
    omtfUnconstrSAMs = cms.InputTag('simOmtfPhase2Digis', 'unconstrCands'),
    
    emtfTracks = cms.InputTag('simEmtfDigisPhase2'),
  
)
