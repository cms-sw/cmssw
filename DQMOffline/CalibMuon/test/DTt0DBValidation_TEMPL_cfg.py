import FWCore.ParameterSet.Config as cms

process = cms.Process("CALIB")

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Geometry.DTGeometry.dtGeometry_cfi")
process.DTGeometryESModule.applyAlignment = False

process.load("DQMServices.Core.DQM_cfg")

process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    #firstRun = cms.untracked.uint32(121475)
    firstRun = cms.untracked.uint32(1)
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
            tag = cms.string('t0'),
            label = cms.untracked.string('tzeroRef')
        )
        ),
    connect = cms.string('sqlite_file:REFT0TEMPLATE'),
    siteLocalConfig = cms.untracked.bool(False)
)

process.t0ToValidate = cms.ESSource("PoolDBESSource",
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(0),
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    timetype = cms.string('runnumber'),
    toGet = cms.VPSet( 
         cms.PSet(
    record = cms.string('DTT0Rcd'),
    tag = cms.string('t0'),
    label = cms.untracked.string('tzeroToValidate')
    )
        ),
    connect = cms.string('sqlite_file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/RUNPERIODTEMPLATE/t0/t0_RUNNUMBERTEMPLATE.db'),
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
        #FwkJob = cms.untracked.PSet(
        #    limit = cms.untracked.int32(0)
        #),
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
    OutputFileName = cms.untracked.string('t0DBMonitoring_RUNNUMBERTEMPLATE.root'),
    labelDB = cms.untracked.string('tzeroToValidate')
)

process.qTester = cms.EDAnalyzer("QualityTester",
    prescaleFactor = cms.untracked.int32(1),
    qtList = cms.untracked.FileInPath('DQMOffline/CalibMuon/data/QualityTests.xml')
)

process.p = cms.Path(process.dtT0Analyzer*process.qTester)
process.DQM.collectorHost = ''


