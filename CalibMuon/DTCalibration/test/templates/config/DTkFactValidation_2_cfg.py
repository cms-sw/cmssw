import FWCore.ParameterSet.Config as cms

process = cms.Process("TTRIGVALIDSUMPROC")

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('resolutionTest_step1',
        'resolutionTest_step2',
        'resolutionTest_step3'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('ERROR'),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        resolution = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        noLineBreaks = cms.untracked.bool(True)
    ),
    categories = cms.untracked.vstring('resolution'),
    destinations = cms.untracked.vstring('cout')
)

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Geometry.DTGeometry.dtGeometry_cfi")
process.DTGeometryESModule.applyAlignment = False

process.load("CondCore.DBCommon.CondDBSetup_cfi")
process.load("DQMServices.Core.DQM_cfg")
#process.load("RecoLocalMuon.Configuration.RecoLocalMuonCosmics_cff")

process.source = cms.Source("EmptyIOVSource",
     lastValue = cms.uint64(100),
     timetype = cms.string('runnumber'),
     firstValue = cms.uint64(1),
     interval = cms.uint64(90)
 )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.eventInfoProvider = cms.EDFilter("EventCoordinatesSource",
    eventInfoFolder = cms.untracked.string('EventInfo/')
)

process.qTester = cms.EDAnalyzer("QualityTester",
    prescaleFactor = cms.untracked.int32(1),
    qtList = cms.untracked.FileInPath('DQM/DTMonitorClient/test/QualityTests_ttrig.xml')
)

process.load("DQM.DTMonitorClient.dtResolutionTest_cfi")
process.resolutionTest.histoTag2D = 'hResDistVsDist_STEP3' 
process.resolutionTest.histoTag  = 'hResDist_STEP3'
process.resolutionTest.STEP = 'STEP3'
process.resolutionTest.OutputMEsInRootFile = cms.bool(True)
process.resolutionTest.readFile = cms.untracked.bool(True)
#process.resolutionTest.inputFile = cms.untracked.string('')
#process.resolutionTest.OutputFileName = cms.string('')

process.secondStep = cms.Sequence(process.resolutionTest*process.qTester)
process.p = cms.Path(process.secondStep)
process.DQM.collectorHost = ''
