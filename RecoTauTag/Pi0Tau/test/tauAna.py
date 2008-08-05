import FWCore.ParameterSet.Config as cms

process = cms.Process("tauAna")
process.load("FWCore.MessageService.MessageLogger_cfi")

process.load("RecoTauTag.Pi0Tau.tauAna_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
#    fileNames = cms.untracked.vstring('file:/uscmst1b_scratch/lpc1/lpctau/dwjang/single_particles/210pre10/stau/pdg15_pt10.0-100.0_events1000_1.root')
    fileNames = cms.untracked.vstring('/store/RelVal/CMSSW_2_1_0/RelValZTT/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V5_v1/0001/0C4FEB17-9E60-DD11-997C-001617E30D00.root')
)

process.tauAna.histFileName = 'hist_ztau.root'

process.p = cms.Path(process.tauAna)

