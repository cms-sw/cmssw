import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.debugModules = cms.untracked.vstring('*')
process.MessageLogger.destinations = cms.untracked.vstring('cerr')
process.MessageLogger.categories.append('resolution')
process.MessageLogger.cerr =  cms.untracked.PSet(
    threshold = cms.untracked.string('DEBUG'),
    noLineBreaks = cms.untracked.bool(False),
    DEBUG = cms.untracked.PSet(limit = cms.untracked.int32(0)),
    INFO = cms.untracked.PSet(limit = cms.untracked.int32(0)),
    resolution = cms.untracked.PSet(limit = cms.untracked.int32(-1))
)

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = ""

process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.load("DQMServices.Core.DQM_cfg")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring()
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.dtunpacker = cms.EDProducer("DTUnpackingModule",
    dataType = cms.string('DDU'),
    inputLabel = cms.InputTag('source'),
    fedbyType = cms.bool(False),
    useStandardFEDid = cms.bool(True),
    dqmOnly = cms.bool(False),                       
    readOutParameters = cms.PSet(
        debug = cms.untracked.bool(False),
        rosParameters = cms.PSet(
            writeSC = cms.untracked.bool(True),
            readingDDU = cms.untracked.bool(True),
            performDataIntegrityMonitor = cms.untracked.bool(False),
            readDDUIDfromDDU = cms.untracked.bool(True),
            debug = cms.untracked.bool(False),
            localDAQ = cms.untracked.bool(False)
        ),
        localDAQ = cms.untracked.bool(False),
        performDataIntegrityMonitor = cms.untracked.bool(False)
    )
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBSetup,
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:t0.db'),
    authenticationMethod = cms.untracked.uint32(0),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('DTT0Rcd'),
        tag = cms.string('t0')
    ))
)

process.eventInfoProvider = cms.EDFilter("EventCoordinatesSource",
    eventInfoFolder = cms.untracked.string('EventInfo/')
)

# test pulse monitoring
process.load("DQM.DTMonitorModule.dtDigiTask_TP_cfi")
process.load("DQM.DTMonitorClient.dtOccupancyTest_TP_cfi")
process.dtTPmonitor.defaultTtrig = 300
process.dtTPmonitor.defaultTmax = 100
process.dtTPmonitor.inTimeHitsLowerBound = 0
process.dtTPmonitor.inTimeHitsUpperBound = 0

process.dtT0WireCalibration = cms.EDAnalyzer("DTT0Calibration",
    # Cells for which you want the histos (default = None)
    cellsWithHisto = cms.untracked.vstring(),
    # Label to retrieve DT digis from the event
    digiLabel = cms.untracked.string('dtunpacker'),
    calibSector = cms.untracked.string('All'),
    # Chose the wheel, sector (default = All)
    calibWheel = cms.untracked.string('All'),
    # Number of events to be used for the t0 per layer histos
    eventsForWireT0 = cms.uint32(25000),
    # Name of the ROOT file which will contain the test pulse times per layer
    rootFileName = cms.untracked.string('DTTestPulses.root'),
    debug = cms.untracked.bool(False),
    rejectDigiFromPeak = cms.uint32(50),
    # Acceptance for TP peak width
    tpPeakWidth = cms.double(15.0),
    # Number of events to be used for the t0 per layer histos
    eventsForLayerT0 = cms.uint32(5000)
)

process.output = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('drop *', 
                                           'keep *_MEtoEDMConverter_*_*'),
    fileName = cms.untracked.string('DQM.root')
)

process.load("DQMServices.Components.MEtoEDMConverter_cff")
process.DQM.collectorHost = ''
"""
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.DQMStore.referenceFileName = ''
process.dqmSaver.convention = 'Offline'
process.dqmSaver.workflow = '/MiniDaq/HIRun2010-v1-dtCalibration-rev1/RAW'
process.DQMStore.collateHistograms = False
process.dqmSaver.convention = "Offline"
"""

process.p = cms.Path(process.dtunpacker*
                     process.dtTPmonitor+process.dtTPmonitorTest+
                     process.dtT0WireCalibration+
                     process.MEtoEDMConverter)
process.outpath = cms.EndPath(process.output)
