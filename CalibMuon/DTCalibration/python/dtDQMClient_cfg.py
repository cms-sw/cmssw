import FWCore.ParameterSet.Config as cms

class config: pass
config.dqmAtRunEnd = False
if config.dqmAtRunEnd: config.fileMode = 'FULLMERGE'
else: config.fileMode = 'NOMERGE'

process = cms.Process("DQMClient")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("DQMServices.Components.EDMtoMEConverter_cff")

process.options = cms.untracked.PSet(
    fileMode = cms.untracked.string(config.fileMode)
)

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Geometry.DTGeometry.dtGeometry_cfi")
process.DTGeometryESModule.applyAlignment = False

process.load("CondCore.DBCommon.CondDBSetup_cfi")
process.load("DQMServices.Core.DQM_cfg")

process.source = cms.Source("PoolSource",
    processingMode = cms.untracked.string("RunsAndLumis"),
    fileNames = cms.untracked.vstring()
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.eventInfoProvider = cms.EDFilter("EventCoordinatesSource",
    eventInfoFolder = cms.untracked.string('EventInfo/')
)

process.qTester = cms.EDAnalyzer("QualityTester",
    prescaleFactor = cms.untracked.int32(1),
    qtList = cms.untracked.FileInPath('CalibMuon/DTCalibration/data/QualityTests_ttrig.xml')
)

process.load("DQM.DTMonitorClient.dtResolutionTestFinalCalib_cfi")

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
                            process.resolutionTest*process.qTester*process.dqmSaver)
process.DQM.collectorHost = ''
