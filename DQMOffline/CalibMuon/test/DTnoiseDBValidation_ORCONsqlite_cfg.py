import FWCore.ParameterSet.Config as cms

process = cms.Process("noiseCALIB")
process.load("CondCore.DBCommon.CondDBSetup_cfi")

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

process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(68958)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.noiseRef = cms.ESSource("PoolDBESSource",
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(0),
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    timetype = cms.string('runnumber'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('DTStatusFlagRcd'),
        tag = cms.string('noise_CRAFT_V01_offline'),
        label = cms.untracked.string('noiseRef')
    ), 
        cms.PSet(
            record = cms.string('DTStatusFlagRcd'),
            tag = cms.string('noise'),
            connect = cms.untracked.string('sqlite_file:noise_66722.db'),
            label = cms.untracked.string('noiseToValidate')
        )),
    connect = cms.string('oracle://cms_orcoff_prod/CMS_COND_30X_DT'),
    siteLocalConfig = cms.untracked.bool(False)                           
)

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('dtNoiseAnalyzer'),
    cout = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        noiseDbValidation = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        noLineBreaks = cms.untracked.bool(True),
        threshold = cms.untracked.string('DEBUG'),
        DEBUG = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        )
    ),
    categories = cms.untracked.vstring('noiseDbValidation'),
    destinations = cms.untracked.vstring('cout')
)

process.dtNoiseAnalyzer = cms.EDAnalyzer("DTnoiseDBValidation",
    labelDBRef = cms.untracked.string('noiseRef'),
    diffTestName = cms.untracked.string('noiseDifferenceInRange'),
    wheelTestName = cms.untracked.string('noiseWheelOccInRange'),
    stationTestName = cms.untracked.string('noiseStationOccInRange'),
    sectorTestName = cms.untracked.string('noiseSectorOccInRange'),                       
    TestName = cms.untracked.string('noiseWheelOccInRange'),                                
    OutputFileName = cms.untracked.string('noiseTestMonitoring.root'),
    labelDB = cms.untracked.string('noiseToValidate')
)

from DQMServices.Core.DQMQualityTester import DQMQualityTester
process.qTester = DQMQualityTester(
    prescaleFactor = cms.untracked.int32(1),
    qtList = cms.untracked.FileInPath('DQMOffline/CalibMuon/data/QualityTests.xml')
)

process.p = cms.Path(process.dtNoiseAnalyzer*process.qTester)
process.DQM.collectorHost = ''


