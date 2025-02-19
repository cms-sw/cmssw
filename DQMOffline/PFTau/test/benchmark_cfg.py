# test file for particle flow DQM Offline.
# creates a root file with histograms filled with PFCandidate data,
# present in the Candidate, and in the PFCandidate classes. 

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")

fa = 'RelValQCD'
fb = 'FlatPt_15_3000_Fast'
fc = 'ParticleFlow'

process.load("RecoParticleFlow.Configuration.DBS_Samples.%s_%s_cfi" % (fa, fb) )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(200)
)

process.load("DQMOffline.PFTau.DQMOfflineParticleFlow_cff")

process.dqmSaver.convention = 'Offline'
process.dqmSaver.workflow = '/%s/%s/%s' % (fa, fb, fc)
process.dqmEnv.subSystemFolder = 'ParticleFlow'

process.p =cms.Path(
    process.DQMOfflineParticleFlowSequence +
    process.dqmEnv +
    process.dqmSaver
    )


process.schedule = cms.Schedule(process.p)


process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 50
