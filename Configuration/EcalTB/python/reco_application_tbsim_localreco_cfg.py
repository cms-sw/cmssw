import FWCore.ParameterSet.Config as cms

process = cms.Process("uncalibRecHitProd")
# initialize  MessageLogger
process.load("FWCore.MessageLogger.MessageLogger_cfi")

# ECAL TBReco sequence 
process.load("Configuration.EcalTB.localReco_tbsim_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:ECALH4TB_detsim_digi.root')
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('ECALH4TB_detsim_hits.root')
)

process.p = cms.Path(process.localReco_tbsim)
process.e = cms.EndPath(process.out)

