# The following comments couldn't be translated into the new config version:

#! /bin/env cmsRun

import FWCore.ParameterSet.Config as cms

process = cms.Process("dqmAnalyzer")
process.load("DQMServices.Components.DQMEnvironment_cfi")

#keep the logging output to a nice level
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("RecoBTag.Configuration.RecoBTag_cff")

process.load("DQMServices.Core.DQM_cfg")

process.load("DQMOffline.RecoB.dqmAnalyzer_cff")

process.load("DQMOffline.RecoB.dqmCollector_cff")

process.load("DQMOffline.RecoB.PrimaryVertexMonitor_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(200)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring()
)

process.plots = cms.Path(process.pvMonitor*process.bTagPlots*process.bTagCollectorSequence*process.dqmSaver)
process.dqmEnv.subSystemFolder = 'BTAG'
process.dqmSaver.producer = 'DQM'
process.dqmSaver.workflow = '/POG/BTAG/BJET'
process.dqmSaver.convention = 'RelVal'
process.PoolSource.fileNames = [
       '/store/relval/CMSSW_2_2_1/RelValTTbar/GEN-SIM-RECO/IDEAL_V9_v2/0002/00E9B0FB-98C4-DD11-AF51-0030487A322E.root',
        '/store/relval/CMSSW_2_2_1/RelValTTbar/GEN-SIM-RECO/IDEAL_V9_v2/0002/28E6A7DA-98C4-DD11-AE6E-001D09F2512C.root'
       ]



