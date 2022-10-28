import FWCore.ParameterSet.Config as cms

process = cms.Process("Validation")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.debugModules = cms.untracked.vstring('')
process.MessageLogger.resolution=dict()
process.MessageLogger.cerr =  cms.untracked.PSet(
    threshold = cms.untracked.string('DEBUG'),
    noLineBreaks = cms.untracked.bool(False),
    DEBUG = cms.untracked.PSet(limit = cms.untracked.int32(0)),
    INFO = cms.untracked.PSet(limit = cms.untracked.int32(0)),
    resolution = cms.untracked.PSet(limit = cms.untracked.int32(-1))
)

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag=autoCond['run3_data']

process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff")
process.load("Geometry.DTGeometry.dtGeometry_cfi")
process.DTGeometryESModule.applyAlignment = False
process.DTGeometryESModule.fromDDD = False

process.load("CondCore.CondDB.CondDB_cfi")
process.load("DQMServices.Core.DQM_cfg")

"""
process.source = cms.Source("EmptyIOVSource",
     lastValue = cms.uint64(100),
     timetype = cms.string('runnumber'),
     firstValue = cms.uint64(1),
     interval = cms.uint64(90)
)
"""
process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.eventInfoProvider = cms.EDFilter("EventCoordinatesSource",
    eventInfoFolder = cms.untracked.string('EventInfo/')
)

from DQMServices.Core.DQMQualityTester import DQMQualityTester
process.qTester = DQMQualityTester(
    prescaleFactor = cms.untracked.int32(1),
    qtList = cms.untracked.FileInPath('CalibMuon/DTCalibration/data/QualityTests_ttrig.xml')
)

process.load("DQM.DTMonitorClient.dtResolutionTest_cfi")
process.resolutionTest.histoTag2D = 'hResDistVsDist_STEP3' 
process.resolutionTest.histoTag  = 'hResDist_STEP3'
process.resolutionTest.STEP = 'STEP3'
process.resolutionTest.OutputMEsInRootFile = cms.bool(True)
process.resolutionTest.readFile = cms.untracked.bool(True)
#process.resolutionTest.inputFile = cms.untracked.string('')
#process.resolutionTest.OutputFileName = cms.string('')

process.dtValidSequence = cms.Sequence(process.resolutionTest*process.qTester)
process.validation_step = cms.Path(process.dtValidSequence)
#process.DQM.collectorHost = ''
