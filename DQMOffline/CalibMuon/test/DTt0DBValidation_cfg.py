import FWCore.ParameterSet.Config as cms

process = cms.Process("CALIB")
process.load("Geometry.MuonCommonData.muonIdealGeometryXML_cfi")

process.load("Geometry.DTGeometry.dtGeometry_cfi")
process.DTGeometryESModule.applyAlignment = False

process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")

process.load("DQMServices.Core.DQM_cfg")

from CalibTracker.Configuration.Common.PoolDBESSource_cfi import poolDBESSource
poolDBESSource.connect = "frontier://FrontierDev/CMS_COND_ALIGNMENT"
poolDBESSource.toGet = cms.VPSet(cms.PSet(
        record = cms.string('GlobalPositionRcd'),
        tag = cms.string('IdealGeometry')
    )) 
process.glbPositionSource = poolDBESSource

process.source = cms.Source("EmptyIOVSource",
    firstValue = cms.uint64(1),
    lastValue = cms.uint64(100),
    timetype = cms.string('runnumber'),
    interval = cms.uint64(90)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.tzeroRef = cms.ESSource("PoolDBESSource",
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(0),
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('DTT0Rcd'),
        tag = cms.string('tzero'),
        connect = cms.untracked.string('sqlite_file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/CRUZET3/t0/t0_CRUZET_080507_1135.db'),
        label = cms.untracked.string('tzeroRef')
    ), 
        cms.PSet(
            record = cms.string('DTT0Rcd'),
            tag = cms.string('tzero'),
            connect = cms.untracked.string('sqlite_file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/CRUZET3/t0/t0_CRUZET_080627_1641.db'),
            label = cms.untracked.string('tzeroToValidate')
        )),
    connect = cms.string('')
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
        FwkJob = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        DEBUG = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        )
    ),
    categories = cms.untracked.vstring('t0dbValidation'),
    destinations = cms.untracked.vstring('cout')
)

process.dtT0Analyzer = cms.EDFilter("DTt0DBValidation",
    labelDBRef = cms.untracked.string('tzeroRef'),
    t0TestName = cms.untracked.string('t0DifferenceInRange'),
    OutputFileName = cms.untracked.string('MuonTestMonitoring_47041.root'),
    labelDB = cms.untracked.string('tzeroToValidate')
)

process.qTester = cms.EDFilter("QualityTester",
    prescaleFactor = cms.untracked.int32(1),
    qtList = cms.untracked.FileInPath('DQMOffline/CalibMuon/data/QualityTests.xml')
)

process.p = cms.Path(process.dtT0Analyzer*process.qTester)
process.DQM.collectorHost = ''


