import FWCore.ParameterSet.Config as cms

class config: pass
config.runNumber = 1
config.refTag = 'DT_noise_cosmic2009_V01_hlt'
config.noiseDB = 'noise.db'
config.dataset = '/SingleMu/Run2011A-DtCalib-v4/ALCARECO'
config.outputdir = 'DQM'
config.trial = 1

# Further config.
dataset_vec = config.dataset.split('/')
config.workflowName = '/%s/%s-dtNoiseDBValidation-rev%d/%s' % (dataset_vec[1],
                                                               dataset_vec[2],
                                                               config.trial,
                                                               dataset_vec[3]) 

process = cms.Process("DBValidation")

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('dtNoiseAnalyzer'),
    cerr = cms.untracked.PSet(
        default = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        DEBUG = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        NoiseDbValidation = cms.untracked.PSet( limit = cms.untracked.int32(10000000) ),
        threshold = cms.untracked.string('DEBUG'),
    ),
    categories = cms.untracked.vstring('NoiseDBValidation'),
    destinations = cms.untracked.vstring('cerr')
)


process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Geometry.DTGeometry.dtGeometry_cfi")
process.DTGeometryESModule.applyAlignment = False

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

process.noiseRef = cms.ESSource("PoolDBESSource",
    process.CondDBSetup,
    connect = cms.string('oracle://cms_orcoff_prod/CMS_COND_31X_DT'),
    toGet = cms.VPSet(
    cms.PSet(
        record = cms.string('DTStatusFlagRcd'),
        tag = cms.string(config.refTag),
        label = cms.untracked.string('noiseRef')
    ), 
    cms.PSet(
        record = cms.string('DTStatusFlagRcd'),
        tag = cms.string('noise'),
        connect = cms.untracked.string('sqlite_file:%s' % config.noiseDB),
        label = cms.untracked.string('noiseToValidate')
    )),
)
process.noiseRef.DBParameters.authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')

process.dtNoiseAnalyzer = cms.EDAnalyzer("DTnoiseDBValidation",
    labelDBRef = cms.string('noiseRef'),
    labelDB = cms.string('noiseToValidate'),
    diffTestName = cms.string('noiseDifferenceInRange'),
    wheelTestName = cms.string('noiseWheelOccInRange'),
    stationTestName = cms.string('noiseStationOccInRange'),
    sectorTestName = cms.string('noiseSectorOccInRange'),                                   
    #OutputFileName = cms.string('noiseDBValidation.root')
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

#process.p = cms.Path(process.dtNoiseAnalyzer*process.qTester*process.dqmSaver)
process.p = cms.Path(process.qTester*
                     process.dtNoiseAnalyzer*
                     process.dqmSaver)

