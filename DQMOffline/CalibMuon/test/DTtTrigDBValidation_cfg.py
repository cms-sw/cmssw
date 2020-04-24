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
    firstRun = cms.untracked.uint32(112281)
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
    toGet = cms.VPSet(
        cms.PSet(
            record = cms.string('DTTtrigRcd'),
            tag = cms.string('ttrig'),
            connect = cms.untracked.string('sqlite_file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/COMM09/ttrig/Run112237/Ttrig/Results/ttrig_ResidCorr_112237.db'), 
            label = cms.untracked.string('ttrigRef')
        ), 
        cms.PSet(
            record = cms.string('DTTtrigRcd'),
            tag = cms.string('ttrig'),
            connect = cms.untracked.string('sqlite_file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/COMM09/ttrig/Run112281/Ttrig/Results/ttrig_ResidCorr_112281.db'),
            label = cms.untracked.string('ttrigToValidate')
        )),
    #connect = cms.string('CMSCONDVSTEMPLATE'),
    connect = cms.string(''),
    siteLocalConfig = cms.untracked.bool(False)
)

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('dtTTrigAnalyzer'),
    cout = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        tTrigdbValidation = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
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
    tTrigTestName = cms.string('tTrigDifferenceInRange')
    #OutputMEsInRootFile = cms.untracked.bool(False),
    #OutputFileName = cms.untracked.string('tTrigDBMonitoring_112281_vs_112237.root')
)

process.qTester = cms.EDAnalyzer("QualityTester",
    prescaleFactor = cms.untracked.int32(1),
    qtList = cms.untracked.FileInPath('DQMOffline/CalibMuon/data/QualityTests.xml')
)

process.load("DQMServices.Components.DQMEnvironment_cfi")
process.dqmSaver.convention = 'Offline'
process.dqmSaver.workflow = '/Muon/DT/DTDBValidation'

process.p = cms.Path(process.dtTTrigAnalyzer*process.qTester*process.dqmSaver)
process.DQM.collectorHost = ''
