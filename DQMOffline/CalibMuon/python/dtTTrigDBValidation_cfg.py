import FWCore.ParameterSet.Config as cms

class config: pass
config.runNumber = 1
config.refTag = 'DTTtrig_V01_prompt'
config.ttrigDB = 'ttrig.db'
config.dataset = '/SingleMu/Run2011A-DtCalib-v4/ALCARECO'
config.outputdir = 'DQM'
config.trial = 1

# Further config.
dataset_vec = config.dataset.split('/')
config.workflowName = '/%s/%s-dtTTrigDBValidation-rev%d/%s' % (dataset_vec[1],
                                                               dataset_vec[2],
                                                               config.trial,
                                                               dataset_vec[3])

process = cms.Process("DBValidation")

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('dtTTrigAnalyzer'),
    cerr = cms.untracked.PSet(
        default = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        DEBUG = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        TTrigDBValidation = cms.untracked.PSet( limit = cms.untracked.int32(10000000) ),
        threshold = cms.untracked.string('DEBUG'),
    ),
    categories = cms.untracked.vstring('TTrigDBValidation'),
    destinations = cms.untracked.vstring('cerr')
)

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Geometry.DTGeometry.dtGeometry_cfi")
process.DTGeometryESModule.applyAlignment = False
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")

process.load("CondCore.DBCommon.CondDBSetup_cfi")
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")

process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(config.runNumber)
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
    connect = cms.string('oracle://cms_orcoff_prod/CMS_COND_31X_DT'),
    #connect = cms.string(''),
    toGet = cms.VPSet(
        cms.PSet(
            record = cms.string('DTTtrigRcd'),
            tag = cms.string(config.refTag),
            label = cms.untracked.string('ttrigRef')
        ), 
        cms.PSet(
            record = cms.string('DTTtrigRcd'),
            tag = cms.string('ttrig'),
            connect = cms.untracked.string('sqlite_file:%s' % config.ttrigDB),
            label = cms.untracked.string('ttrigToValidate')
        )
    ),
    siteLocalConfig = cms.untracked.bool(False)
)

process.dtTTrigAnalyzer = cms.EDAnalyzer("DTtTrigDBValidation",
    labelDBRef = cms.string('ttrigRef'),
    labelDB = cms.string('ttrigToValidate'),
    tTrigTestName = cms.string('tTrigDifferenceInRange'),
    #OutputFileName = cms.string('tTrigDBValidation_DT_tTrig_cosmics_2009_v3_prompt.root')
)

process.qTester = cms.EDAnalyzer("QualityTester",
    prescaleFactor = cms.untracked.int32(1),
    reportThreshold = cms.untracked.string('black'),
    qtList = cms.untracked.FileInPath('DQMOffline/CalibMuon/data/QualityTests.xml')
)

process.DQMStore.referenceFileName = ''
process.dqmSaver.convention = 'Offline'
process.dqmSaver.workflow = config.workflowName
process.dqmSaver.dirName = config.outputdir
process.DQMStore.collateHistograms = False
process.DQM.collectorHost = ''
"""
process.DQMStore.referenceFileName = ''
process.DQMStore.collateHistograms = True
process.dqmSaver.convention = 'Offline'
process.dqmSaver.workflow = workflowName
process.dqmSaver.saveByRun = -1
process.dqmSaver.saveAtJobEnd = True
process.dqmSaver.forceRunNumber = runNumber
"""

#process.p = cms.Path(process.dtTTrigAnalyzer*process.qTester*process.dqmSaver)
process.p = cms.Path(process.qTester*
                     process.dtTTrigAnalyzer*
                     process.dqmSaver)
