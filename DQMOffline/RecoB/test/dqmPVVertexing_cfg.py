# The following comments couldn't be translated into the new config version:

#! /bin/env cmsRun

import FWCore.ParameterSet.Config as cms

process = cms.Process("dqmAnalyzer")
process.load("DQMServices.Components.DQMEnvironment_cfi")

#keep the logging output to a nice level
process.load("FWCore.MessageLogger.MessageLogger_cfi")


process.load("DQMServices.Core.DQM_cfg")

process.load("DQMOffline.RecoB.PrimaryVertexMonitor_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring()
)


process.plots = cms.Path(process.pvMonitor*process.dqmSaver)
process.dqmEnv.subSystemFolder = 'BTAG'
process.dqmSaver.producer = 'DQM'
process.dqmSaver.workflow = '/POG/BTAG/PV'
process.dqmSaver.convention = 'Offline'
process.PoolSource.fileNames = ['/store/relval/CMSSW_3_3_3/RelValTTbar/GEN-SIM-RECO/MC_31X_V9-v2/0002/D84DEA71-0CD2-DE11-AB3C-002618943958.root']


