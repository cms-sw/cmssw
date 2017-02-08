import FWCore.ParameterSet.Config as cms

process = cms.Process("EDMtoMEConvert")
process.load("DQMServices.Components.EDMtoMEConverter_cff")
process.options = cms.untracked.PSet(
 fileMode = cms.untracked.string('FULLMERGE')
)

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Geometry.DTGeometry.dtGeometry_cfi")
process.DTGeometryESModule.applyAlignment = False
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
process.load("CondCore.DBCommon.CondDBSetup_cfi")
process.load("DQMServices.Core.DQM_cfg")

process.source = cms.Source("PoolSource",
    processingMode = cms.untracked.string("RunsLumisAndEvents"),
    fileNames = cms.untracked.vstring(
	'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/LHC/Temp/ttrig/Run121475/Ttrig/Validation/crab_0_091123_221340/res/DQM_1.root',
	'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/LHC/Temp/ttrig/Run121475/Ttrig/Validation/crab_0_091123_221340/res/DQM_10.root',
	'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/LHC/Temp/ttrig/Run121475/Ttrig/Validation/crab_0_091123_221340/res/DQM_11.root',
	'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/LHC/Temp/ttrig/Run121475/Ttrig/Validation/crab_0_091123_221340/res/DQM_12.root',
	'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/LHC/Temp/ttrig/Run121475/Ttrig/Validation/crab_0_091123_221340/res/DQM_13.root',
	'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/LHC/Temp/ttrig/Run121475/Ttrig/Validation/crab_0_091123_221340/res/DQM_14.root',
	'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/LHC/Temp/ttrig/Run121475/Ttrig/Validation/crab_0_091123_221340/res/DQM_15.root',
	'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/LHC/Temp/ttrig/Run121475/Ttrig/Validation/crab_0_091123_221340/res/DQM_16.root',
	'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/LHC/Temp/ttrig/Run121475/Ttrig/Validation/crab_0_091123_221340/res/DQM_17.root',
	'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/LHC/Temp/ttrig/Run121475/Ttrig/Validation/crab_0_091123_221340/res/DQM_18.root',
	'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/LHC/Temp/ttrig/Run121475/Ttrig/Validation/crab_0_091123_221340/res/DQM_2.root',
	'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/LHC/Temp/ttrig/Run121475/Ttrig/Validation/crab_0_091123_221340/res/DQM_20.root',
	'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/LHC/Temp/ttrig/Run121475/Ttrig/Validation/crab_0_091123_221340/res/DQM_21.root',
	'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/LHC/Temp/ttrig/Run121475/Ttrig/Validation/crab_0_091123_221340/res/DQM_22.root',
	'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/LHC/Temp/ttrig/Run121475/Ttrig/Validation/crab_0_091123_221340/res/DQM_3.root',
	'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/LHC/Temp/ttrig/Run121475/Ttrig/Validation/crab_0_091123_221340/res/DQM_4.root',
	'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/LHC/Temp/ttrig/Run121475/Ttrig/Validation/crab_0_091123_221340/res/DQM_5.root',
	'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/LHC/Temp/ttrig/Run121475/Ttrig/Validation/crab_0_091123_221340/res/DQM_6.root',
	'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/LHC/Temp/ttrig/Run121475/Ttrig/Validation/crab_0_091123_221340/res/DQM_7.root',
	'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/LHC/Temp/ttrig/Run121475/Ttrig/Validation/crab_0_091123_221340/res/DQM_8.root',
	'file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/LHC/Temp/ttrig/Run121475/Ttrig/Validation/crab_0_091123_221340/res/DQM_9.root'


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
        resolution = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        noLineBreaks = cms.untracked.bool(True)
    ),
    categories = cms.untracked.vstring('resolution'),
    destinations = cms.untracked.vstring('cout')
)

process.qTester = cms.EDAnalyzer("QualityTester",
    prescaleFactor = cms.untracked.int32(1),
    qtList = cms.untracked.FileInPath('DQM/DTMonitorClient/test/QualityTests_ttrig.xml')
)

process.load("DQM.DTMonitorClient.dtResolutionTestFinalCalib_cfi")
process.modulo=process.resolutionTest.clone()

process.source.processingMode = "RunsAndLumis"
process.DQMStore.referenceFileName = ''
process.dqmSaver.convention = 'Offline'
process.dqmSaver.workflow = '/Muon/Dt/Test1'
process.DQMStore.collateHistograms = False
process.EDMtoMEConverter.convertOnEndLumi = True
process.EDMtoMEConverter.convertOnEndRun = False

process.p = cms.Path(process.EDMtoMEConverter*process.modulo*process.qTester*process.dqmSaver)
process.DQM.collectorHost = ''

