import FWCore.ParameterSet.Config as cms

class config: pass
config.dqmAtRunEnd = False
if config.dqmAtRunEnd: config.fileMode = 'FULLMERGE'
else: config.fileMode = 'NOMERGE'

process = cms.Process("DQMClient")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.debugModules = cms.untracked.vstring('')
process.MessageLogger.DTDQM=dict()
process.MessageLogger.resolution=dict()
process.MessageLogger.cerr =  cms.untracked.PSet(
    threshold = cms.untracked.string('WARNING'),
    noLineBreaks = cms.untracked.bool(False),
    DEBUG = cms.untracked.PSet(limit = cms.untracked.int32(0)),
    INFO = cms.untracked.PSet(limit = cms.untracked.int32(0)),
    DTDQM = cms.untracked.PSet(limit = cms.untracked.int32(-1)), 
    resolution = cms.untracked.PSet(limit = cms.untracked.int32(-1))
)

process.load("DQMServices.Components.EDMtoMEConverter_cff")

process.options = cms.untracked.PSet(
    fileMode = cms.untracked.string(config.fileMode)
)

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = ''

process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff")
process.load("Geometry.DTGeometry.dtGeometry_cfi")
process.DTGeometryESModule.applyAlignment = False
process.DTGeometryESModule.fromDDD = False

process.load("CondCore.CondDB.CondDB_cfi")
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

from DQMServices.Core.DQMQualityTester import DQMQualityTester
process.qTester = DQMQualityTester(
    prescaleFactor = cms.untracked.int32(1),
    qtList = cms.untracked.FileInPath('CalibMuon/DTCalibration/data/QualityTests_ttrig.xml')
)

#process.load("DQM.DTMonitorClient.dtResolutionTestFinalCalib_cfi")
process.load("DQM.DTMonitorClient.dtResolutionTest_cfi")
process.resolutionTest.calibModule = True
process.resolutionTest.histoTag2D = 'hResDistVsDist_STEP3'
process.resolutionTest.histoTag  = 'hResDist_STEP3'
process.resolutionTest.STEP = 'STEP3'
process.resolutionTest.meanMaxLimit = 0.02
process.resolutionTest.sigmaTest = True
process.resolutionTest.slopeTest = False
process.resolutionTest.meanWrongHisto = cms.untracked.bool(False)
process.resolutionTest.sigmaWrongHisto = cms.untracked.bool(False)
process.resolutionTest.readFile = cms.untracked.bool(False) 
process.resolutionTest.OutputMEsInRootFile = cms.bool(False)
#process.resolutionTest.inputFile = cms.untracked.string('')
#process.resolutionTest.OutputFileName = cms.string('')

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

#process.dqm_step = cms.Path(process.EDMtoMEConverter*
#                            process.qTester*process.resolutionTest*process.dqmSaver)
process.dqm_step = cms.Path(process.EDMtoMEConverter*
                            process.dqmSaver)

#process.DQM.collectorHost = ''
