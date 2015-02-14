import FWCore.ParameterSet.Config as cms

process = cms.Process("CALIB")

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Geometry.DTGeometry.dtGeometry_cfi")
process.DTGeometryESModule.applyAlignment = False

process.load("DQMServices.Core.DQM_cfg")

process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(111873)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.tzeroRef = cms.ESSource("PoolDBESSource",
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(0),
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    timetype = cms.string('runnumber'),
    toGet = cms.VPSet(
        cms.PSet(
            record = cms.string('DTT0Rcd'),
            tag = cms.string('t0_CRAFT_V01_offline'),
            label = cms.untracked.string('tzeroRef')
        ), 
        cms.PSet(
            record = cms.string('DTT0Rcd'),
            tag = cms.string('t0'),
            connect = cms.untracked.string('sqlite_file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/COMM09/t0/t0_111873.db'),
            label = cms.untracked.string('tzeroToValidate')
        ) 
    ),
    connect = cms.string('frontier://FrontierProd/CMS_COND_31X_FROM21X'),
    siteLocalConfig = cms.untracked.bool(False)
)

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('dtT0Analyzer'),
    cout = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        t0dbValidation = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        noLineBreaks = cms.untracked.bool(True),
        threshold = cms.untracked.string('DEBUG'),
        DEBUG = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        )
    ),
    categories = cms.untracked.vstring('t0dbValidation'),
    destinations = cms.untracked.vstring('cout')
)

process.dtT0Analyzer = cms.EDAnalyzer("DTt0DBValidation",
    labelDBRef = cms.untracked.string('tzeroRef'),
    t0TestName = cms.untracked.string('t0DifferenceInRange'),
    OutputFileName = cms.untracked.string('t0TestMonitoring_111873.root'),
    labelDB = cms.untracked.string('tzeroToValidate')
)

process.qTester = cms.EDAnalyzer("QualityTester",
    prescaleFactor = cms.untracked.int32(1),
    qtList = cms.untracked.FileInPath('DQMOffline/CalibMuon/data/QualityTests.xml')
)

process.p = cms.Path(process.dtT0Analyzer*process.qTester)
process.DQM.collectorHost = ''
