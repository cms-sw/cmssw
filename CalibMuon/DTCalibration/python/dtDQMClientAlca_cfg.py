import FWCore.ParameterSet.Config as cms

class config: pass
config.dqmAtRunEnd = True
if config.dqmAtRunEnd: config.fileMode = 'FULLMERGE'
else: config.fileMode = 'NOMERGE'

process = cms.Process("HARVESTING")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag=autoCond['run3_data']

process.load("CondCore.CondDB.CondDB_cfi")
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
    process.dqmSaver.convention = 'Offline'
    process.dqmSaver.workflow = workflowName
    process.EDMtoMEConverter.convertOnEndLumi = True
    process.EDMtoMEConverter.convertOnEndRun = True
else:
    process.dqmSaver.convention = 'Offline'
    process.dqmSaver.workflow = workflowName
    process.EDMtoMEConverter.convertOnEndLumi = True
    process.EDMtoMEConverter.convertOnEndRun = True
    process.dqmSaver.saveByRun = -1
    process.dqmSaver.saveAtJobEnd = True  
    process.dqmSaver.forceRunNumber = 1

process.dqm_step = cms.Path(process.EDMtoMEConverter*
                            process.ALCARECODTCalibSynchDQMClient*process.dqmSaver)
#process.DQM.collectorHost = ''
