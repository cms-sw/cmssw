import FWCore.ParameterSet.Config as cms

# AlCaReco for muon based alignment using ZMuMu events
OutCSA06ZMuMu_muon = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathCSA06ZMuMu_muon')
    ),
    outputCommands = cms.untracked.vstring('drop *', 'keep *_dt1DRecHits_*_*', 'keep *_dt2DSegments_*_*', 'keep *_dt4DSegments_*_*', 'keep *_csc2DRecHits_*_*', 'keep *_cscSegments_*_*', 'keep *_rpcRecHits_*_*', 'keep *_standAloneMuons_*_*')
)

