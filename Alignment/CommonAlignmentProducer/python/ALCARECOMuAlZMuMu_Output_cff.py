import FWCore.ParameterSet.Config as cms

# AlCaReco for muon based alignment using ZMuMu events
OutALCARECOMuAlZMuMu = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOMuAlZMuMu')
    ),
    outputCommands = cms.untracked.vstring('drop *', 'keep *_ALCARECOMuAlZMuMu_*_*', 'keep *_muonCSCDigis_*_*', 'keep *_muonDTDigis_*_*', 'keep *_muonRPCDigis_*_*', 'keep *_dt1DRecHits_*_HLT', 'keep *_dt2DSegments_*_HLT', 'keep *_dt4DSegments_*_HLT', 'keep *_csc2DRecHits_*_HLT', 'keep *_cscSegments_*_HLT', 'keep *_rpcRecHits_*_HLT')
)

