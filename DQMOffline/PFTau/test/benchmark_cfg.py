# test file for particle flow DQM Offline.
# creates a root file with histograms filled with PFCandidate data,
# present in the Candidate, and in the PFCandidate classes. 

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")

process.load("RecoParticleFlow.Configuration.DBS_Samples.RelValQCD_FlatPt_15_3000_Fast_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.load("DQMOffline.PFTau.pfCandidateBenchmark_cff")

process.dqmSaver.convention = 'Offline'
process.dqmSaver.workflow = '/A/B/C'
process.dqmEnv.subSystemFolder = 'ParticleFlow'

process.p =cms.Path(
    process.pfCandidateBenchmarkSequence +
    process.dqmEnv +
    process.dqmSaver
    )


process.schedule = cms.Schedule(process.p)


process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 50
