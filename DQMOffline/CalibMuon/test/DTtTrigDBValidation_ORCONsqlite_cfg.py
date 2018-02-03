import FWCore.ParameterSet.Config as cms

process = cms.Process("CALIB")

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Geometry.DTGeometry.dtGeometry_cfi")
process.DTGeometryESModule.applyAlignment = False
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")

process.load("CondCore.DBCommon.CondDBSetup_cfi")
process.load("DQMServices.Core.DQM_cfg")

process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(70675)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.ttrigRef = cms.ESSource("PoolDBESSource",
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(0),
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    timetype = cms.string('runnumber'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('DTTtrigRcd'),
        tag = cms.string('DT_tTrig_CRAFT_V04_k-07_offline'),
        label = cms.untracked.string('ttrigRef')
    ), 
        cms.PSet(
            record = cms.string('DTTtrigRcd'),
            tag = cms.string('ttrig'),
            connect = cms.untracked.string('sqlite_file:ttrig_67838.db'),
            label = cms.untracked.string('ttrigToValidate')
        )),
    connect = cms.string('oracle://cms_orcoff_prep/CMS_COND_30X_DT'),
    siteLocalConfig = cms.untracked.bool(False)
)

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('dtTTrigAnalyzer'),
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
    categories = cms.untracked.vstring('tTrigdbValidation'),
    destinations = cms.untracked.vstring('cout')
)

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
process.dtTTrigAnalyzer = DQMEDAnalyzer('DTtTrigDBValidation',
    labelDBRef = cms.string('ttrigRef'),
    labelDB = cms.string('ttrigToValidate'),
    tTrigTestName = cms.string('tTrigDifferenceInRange'),
    OutputMEsInRootFile = cms.untracked.bool(True),
    OutputFileName = cms.untracked.string('tTrigTestMonitoring.root')
)

process.qTester = cms.EDAnalyzer("QualityTester",
    prescaleFactor = cms.untracked.int32(1),
    qtList = cms.untracked.FileInPath('DQMOffline/CalibMuon/data/QualityTests.xml')
)

process.p = cms.Path(process.dtTTrigAnalyzer*process.qTester)
process.DQM.collectorHost = ''


