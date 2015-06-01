import FWCore.ParameterSet.Config as cms

class config: pass
config.dqmAtRunEnd = True
if config.dqmAtRunEnd: config.fileMode = 'FULLMERGE'
else: config.fileMode = 'NOMERGE'

process = cms.Process("HARVESTING")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff")
process.GlobalTag.globaltag = ""

process.load("CondCore.DBCommon.CondDBSetup_cfi")
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.EDMtoMEConverter_cff")

process.options = cms.untracked.PSet(
    fileMode = cms.untracked.string(config.fileMode)
)

process.source = cms.Source("PoolSource",
    processingMode = cms.untracked.string("RunsAndLumis"),
    fileNames = cms.untracked.vstring()
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.load('DQM.DTMonitorClient.ALCARECODTCalibSynchDQMClient_cff')

workflowName = '/Mu/Calibration-v1/DQM'
if config.dqmAtRunEnd:
    process.DQMStore.referenceFileName = ''
    process.dqmSaver.convention = 'Offline'
    process.dqmSaver.workflow = workflowName
    process.DQMStore.collateHistograms = False
    process.EDMtoMEConverter.convertOnEndLumi = True
    process.EDMtoMEConverter.convertOnEndRun = True
else:
    process.DQMStore.referenceFileName = ''
    process.dqmSaver.convention = 'Offline'
    process.dqmSaver.workflow = workflowName
    process.DQMStore.collateHistograms = True
    process.EDMtoMEConverter.convertOnEndLumi = True
    process.EDMtoMEConverter.convertOnEndRun = True
    process.dqmSaver.saveByRun = -1
    process.dqmSaver.saveAtJobEnd = True  
    process.dqmSaver.forceRunNumber = 1

process.dqm_step = cms.Path(process.EDMtoMEConverter*
                            process.ALCARECODTCalibSynchDQMClient*process.dqmSaver)
#process.DQM.collectorHost = ''
