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

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(20)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring()
)

process.plots = cms.Path(process.bTagPlots*process.bTagCollectorSequence*process.dqmSaver)
process.dqmEnv.subSystemFolder = 'BTAG'
process.dqmSaver.producer = 'DQM'
process.dqmSaver.workflow = '/POG/BTAG/BJET'
process.dqmSaver.convention = 'RelVal'
process.PoolSource.fileNames = ['/store/relvall/2008/5/4/RelVal-RelValTTbar-1209247429-IDEAL_V1-2nd/0000/044390E5-EF19-DD11-BD58-000423D98DC4.root', '/store/relvall/2008/5/4/RelVal-RelValTTbar-1209247429-IDEAL_V1-2nd/0000/20E9690D-F019-DD11-9DF6-001617E30D38.root', '/store/relvall/2008/5/4/RelVal-RelValTTbar-1209247429-IDEAL_V1-2nd/0000/3C694447-3F1A-DD11-9766-0016177CA7A0.root']


