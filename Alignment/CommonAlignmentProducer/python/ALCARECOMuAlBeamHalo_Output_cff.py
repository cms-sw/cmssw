import FWCore.ParameterSet.Config as cms

# AlCaReco for muon based alignment using beam-halo muons in the CSC overlap regions
OutALCARECOMuAlBeamHalo = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOMuAlBeamHalo')
    ),
    outputCommands = cms.untracked.vstring('drop *', 'keep *_ALCARECOMuAlBeamHalo_*_*', 'keep *_muonCSCDigis_*_*', 'keep *_csc2DRecHits_*_*', 'keep *_cscSegments_*_*')
)

