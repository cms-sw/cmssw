import FWCore.ParameterSet.Config as cms

process = cms.Process("tauAna")
process.load("FWCore.MessageService.MessageLogger_cfi")

process.load("RecoTauTag.Pi0Tau.tauAna_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:/uscmst1b_scratch/lpc1/lpctau/dwjang/single_particles/210pre10/stau/pdg15_pt10.0-100.0_events1000_1.root')
)

process.tauAna.histFileName = 'hist_stau.root'

process.p = cms.Path(process.tauAna)

