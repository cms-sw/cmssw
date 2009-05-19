import FWCore.ParameterSet.Config as cms

process = cms.Process("EDMtoMEConvert")
process.load("DQMServices.Components.EDMtoMEConverter_cff")
process.options = cms.untracked.PSet(
 fileMode = cms.untracked.string('FULLMERGE')
)


process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Geometry.DTGeometry.dtGeometry_cfi")
process.DTGeometryESModule.applyAlignment = False
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
process.load("CondCore.DBCommon.CondDBSetup_cfi")
process.load("DQMServices.Core.DQM_cfg")
process.load("RecoLocalMuon.Configuration.RecoLocalMuonCosmics_cff")


process.source = cms.Source("PoolSource",
    processingMode = cms.untracked.string("RunsLumisAndEvents"),
    fileNames = cms.untracked.vstring(
	'file:/data/maselli/31X/Run67838/Ttrig/Validation/crab_0_090515_132857/res/DQM_10.root',
	'file:/data/maselli/31X/Run67838/Ttrig/Validation/crab_0_090515_132857/res/DQM_11.root',
	'file:/data/maselli/31X/Run67838/Ttrig/Validation/crab_0_090515_132857/res/DQM_12.root',
	'file:/data/maselli/31X/Run67838/Ttrig/Validation/crab_0_090515_132857/res/DQM_13.root',
	'file:/data/maselli/31X/Run67838/Ttrig/Validation/crab_0_090515_132857/res/DQM_14.root'
    )
)


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.eventInfoProvider = cms.EDFilter("EventCoordinatesSource",
    eventInfoFolder = cms.untracked.string('EventInfo/')
)

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('resolutionTest_step1', 
        'resolutionTest_step2', 
        'resolutionTest_step3'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('ERROR'),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        FwkJob = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        resolution = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        noLineBreaks = cms.untracked.bool(True)
    ),
    categories = cms.untracked.vstring('resolution'),
    destinations = cms.untracked.vstring('cout')
)

process.qTester = cms.EDFilter("QualityTester",
    prescaleFactor = cms.untracked.int32(1),
    qtList = cms.untracked.FileInPath('DQM/DTMonitorClient/test/QualityTests_ttrig.xml')
)

process.load("DQM.DTMonitorClient.dtResolutionTest_cfi")
process.modulo1=process.resolutionTest.clone()
process.modulo1.histoTag2D = 'hResDistVsDist_STEP1' 
process.modulo1.histoTag  = 'hResDist_STEP1'
process.modulo1.STEP = 'STEP1'

process.load("DQM.DTMonitorClient.dtResolutionTest_cfi")
process.modulo2=process.resolutionTest.clone()
process.modulo2.histoTag2D = 'hResDistVsDist_STEP2' 
process.modulo2.histoTag  = 'hResDist_STEP2'
process.modulo2.STEP = 'STEP2'

process.load("DQM.DTMonitorClient.dtResolutionTest_cfi")
process.modulo3=process.resolutionTest.clone()
process.modulo3.histoTag2D = 'hResDistVsDist_STEP3' 
process.modulo3.histoTag  = 'hResDist_STEP3'
process.modulo3.STEP = 'STEP3'

process.source.processingMode = "RunsAndLumis"
process.DQMStore.referenceFileName = ''
process.dqmSaver.convention = 'Offline'
process.dqmSaver.workflow = '/Muon/Dt/Test1'
process.DQMStore.collateHistograms = False
process.EDMtoMEConverter.convertOnEndLumi = True
process.EDMtoMEConverter.convertOnEndRun = False

process.p = cms.Path(process.EDMtoMEConverter*process.modulo1*process.modulo2*process.modulo3*process.qTester*process.dqmSaver)
process.DQM.collectorHost = ''

