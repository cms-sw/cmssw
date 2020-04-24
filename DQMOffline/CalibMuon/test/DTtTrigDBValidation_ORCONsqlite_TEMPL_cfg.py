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
    firstRun = cms.untracked.uint32(RUNNUMBERTEMPLATE)
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
        tag = cms.string('REFTTRIGTEMPLATE'),
        label = cms.untracked.string('ttrigRef')
    ), 
        cms.PSet(
            record = cms.string('DTTtrigRcd'),
            tag = cms.string('ttrig'),
            connect = cms.untracked.string('sqlite_file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/RUNPERIODTEMPLATE/ttrig/ttrig_ResidCorr_RUNNUMBERTEMPLATE.db'),
            label = cms.untracked.string('ttrigToValidate')
        )),
    connect = cms.string('CMSCONDVSTEMPLATE'),
    siteLocalConfig = cms.untracked.bool(False)
)

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('dtTTrigAnalyzer'),
    cout = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        tTrigdbValidation = cms.untracked.PSet(
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

process.dtTTrigAnalyzer = cms.EDAnalyzer("DTtTrigDBValidation",
    labelDBRef = cms.string('ttrigRef'),
    labelDB = cms.string('ttrigToValidate'),
    tTrigTestName = cms.string('tTrigDifferenceInRange'),
    OutputMEsInRootFile = cms.untracked.bool(True),
    OutputFileName = cms.untracked.string('tTrigDBMonitoring_RUNNUMBERTEMPLATE.root')
)

process.qTester = cms.EDAnalyzer("QualityTester",
    prescaleFactor = cms.untracked.int32(1),
    qtList = cms.untracked.FileInPath('DQMOffline/CalibMuon/data/QualityTests.xml')
)

process.p = cms.Path(process.dtTTrigAnalyzer*process.qTester)
process.DQM.collectorHost = ''
