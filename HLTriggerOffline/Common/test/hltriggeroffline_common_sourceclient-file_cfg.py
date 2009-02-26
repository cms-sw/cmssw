import FWCore.ParameterSet.Config as cms

process = cms.Process("DQM")
#
#  DQM SERVICES
#
process.load("DQMServices.Core.DQM_cfg")

#
#  DQM SOURCES
#
process.load("CondCore.DBCommon.CondDBSetup_cfi")
#process.load("Configuration.GlobalRuns.ForceZeroTeslaField_cff")
#process.load("Configuration.GlobalRuns.ReconstructionGR_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("DQMServices.Components.DQMEnvironment_cfi")

process.load("HLTriggerOffline.Common.FourVectorHLTriggerOffline_cfi")
process.load("HLTriggerOffline.Common.FourVectorHLTriggerOfflineClient_cfi")
process.load("Geometry.CaloEventSetup.CaloTopology_cfi")
#process.load("HLTrigger.HLTcore.triggerSummaryAnalyzerAOD_cfi")


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)
process.source = cms.Source("PoolSource",
    fileNames = 
cms.untracked.vstring('file:test.root')
#cms.untracked.vstring('/store/relval/CMSSW_3_1_0_pre1/RelValMinBias/GEN-SIM-RECO/STARTUP_30X_v1/0001/48C7FFEB-49F4-DD11-85AB-001617C3B79A.root')
#cms.untracked.vstring('/store/relval/CMSSW_3_1_0_pre1/RelValSingleMuPt100/GEN-SIM-RECO/IDEAL_30X_v1/0001/325025BD-49F4-DD11-92D4-001617C3B70E.root')
)

process.MessageLogger = cms.Service("MessageLogger",
    detailedInfo = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG')
    ),
    critical = cms.untracked.PSet(
        threshold = cms.untracked.string('ERROR')
    ),
    debugModules = cms.untracked.vstring('hltResults'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING'),
        WARNING = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        noLineBreaks = cms.untracked.bool(True)
    ),
    destinations = cms.untracked.vstring('detailedInfo', 
        'critical', 
        'cout')
)

#process.psource = cms.Path(process.hltResults*process.triggerSummaryAnalyzerAOD)
process.psource = cms.Path(process.hltResults)
process.p = cms.EndPath(process.dqmSaver)
process.DQMStore.verbose = 0
process.DQM.collectorHost = ''
process.dqmSaver.convention = 'Online'
process.dqmSaver.saveByRun = 1
process.dqmSaver.saveAtJobEnd = True


