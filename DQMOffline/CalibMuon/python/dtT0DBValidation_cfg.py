import FWCore.ParameterSet.Config as cms

class config: pass
config.runNumber = 1
config.refTag = 'DT_t0_cosmic2009_V01_express'
config.t0DB = 't0.db'
config.dataset = '/MiniDaq/Run2011A-v1/RAW'
config.outputdir = 'DQM'
config.trial = 1

# Further config.
dataset_vec = config.dataset.split('/')
config.workflowName = '/%s/%s-dtT0DBValidation-rev%d/%s' % (dataset_vec[1],
                                                            dataset_vec[2],
                                                            config.trial,
                                                            dataset_vec[3])

process = cms.Process("DBValidation")

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('dtT0Analyzer'),
    cerr = cms.untracked.PSet(
        default = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        InterChannelSynchDBValidation = cms.untracked.PSet( limit = cms.untracked.int32(10000000) ),
        threshold = cms.untracked.string('DEBUG'),
        DEBUG = cms.untracked.PSet( limit = cms.untracked.int32(0) )
    ),
    categories = cms.untracked.vstring('InterChannelSynchDBValidation'),
    destinations = cms.untracked.vstring('cerr')
)

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Geometry.DTGeometry.dtGeometry_cfi")
process.DTGeometryESModule.applyAlignment = False

process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")

process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(config.runNumber)
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
    connect = cms.string('oracle://cms_orcoff_prod/CMS_COND_31X_DT'),
    toGet = cms.VPSet(
        cms.PSet(
            record = cms.string('DTT0Rcd'),
            tag = cms.string(config.refTag),
            label = cms.untracked.string('tzeroRef')
        ), 
        cms.PSet(
            record = cms.string('DTT0Rcd'),
            tag = cms.string('t0'),
            connect = cms.untracked.string('sqlite_file:%s' % config.t0DB),
            label = cms.untracked.string('tzeroToValidate')
        ) 
    ),
    siteLocalConfig = cms.untracked.bool(False)
)

process.dtT0Analyzer = cms.EDAnalyzer("DTt0DBValidation",
    labelDBRef = cms.string('tzeroRef'),
    labelDB = cms.string('tzeroToValidate'),
    t0TestName = cms.string('t0DifferenceInRange'),
    #OutputFileName = cms.untracked.string('t0DBValidation_DT_t0_cosmic2009_V01_express.root')
)

process.qTester = cms.EDAnalyzer("QualityTester",
    prescaleFactor = cms.untracked.int32(1),
    reportThreshold = cms.untracked.string('black'),
    qtList = cms.untracked.FileInPath('DQMOffline/CalibMuon/data/QualityTests.xml')
)

process.DQMStore.referenceFileName = ''
process.dqmSaver.convention = 'Offline'
process.dqmSaver.workflow = config.workflowName
process.DQMStore.collateHistograms = False
process.DQM.collectorHost = ''

#process.p = cms.Path(process.dtT0Analyzer*process.qTester*process.dqmSaver)
process.p = cms.Path(process.qTester*
                     process.dtT0Analyzer*
                     process.dqmSaver)
