import FWCore.ParameterSet.Config as cms

gmtFwdMuons = cms.EDProducer(
    'Phase2L1TGMTFwdMuonTranslator',
    muons      = cms.InputTag('simGmtStage2Digis'),
    stubs      = cms.InputTag('gmtStubs','tps'),
    emtfTracks = cms.InputTag('simEmtfDigisPhase2'),
)
