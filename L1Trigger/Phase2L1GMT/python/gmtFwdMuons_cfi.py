import FWCore.ParameterSet.Config as cms

gmtFwdMuons = cms.EDProducer(
    'Phase2L1TGMTFwdMuonTranslator',
    stubs      = cms.InputTag('gmtStubs','tps'),
    omtfTracks = cms.InputTag('simOmtfPhase2Digis', 'OMTF'),
    emtfTracks = cms.InputTag('simEmtfDigisPhase2'),
  
)
