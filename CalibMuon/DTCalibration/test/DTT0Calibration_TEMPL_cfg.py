import FWCore.ParameterSet.Config as cms

import sys

#filename = None
filename = '/afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/RUNPERIODTEMPL/t0/tpDead_reference.txt'
for opt in sys.argv:
    if opt[:7] == 'tpDead=': filename = opt[7:]
 
process = cms.Process("PROD")

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "GLOBALTAGTEMPLATE"

process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.load("DQMServices.Core.DQM_cfg")

process.source = cms.Source("PoolSource",
#    debugFlag = cms.untracked.bool(True),
#    debugVebosity = cms.untracked.uint32(10),
    fileNames = cms.untracked.vstring()
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(800000)
)

process.dtunpacker = cms.EDProducer("DTUnpackingModule",
    dataType = cms.string('DDU'),
    useStandardFEDid = cms.untracked.bool(True),
    fedbyType = cms.untracked.bool(True),
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
    DBParameters = cms.PSet(
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    authenticationMethod = cms.untracked.uint32(0),
    connect = cms.string('sqlite_file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/RUNPERIODTEMPL/t0/t0_RUNNUMBERTEMPLATE.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('DTT0Rcd'),
        tag = cms.string('t0')
    ))
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
        #FwkJob = cms.untracked.PSet(
        #    limit = cms.untracked.int32(0)
        #),
        resolution = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        noLineBreaks = cms.untracked.bool(True)
    ),
    categories = cms.untracked.vstring('resolution'),
    destinations = cms.untracked.vstring('cout')
)

# test pulse monitoring
process.load("DQM.DTMonitorModule.dtDigiTask_TP_cfi")
process.load("DQM.DTMonitorClient.dtOccupancyTest_TP_cfi")
#from DQM.DTMonitorModule.dtDigiTask_TP_cfi import *
#from DQM.DTMonitorClient.dtOccupancyTest_TP_cfi import *
process.dtTPmonitor.defaultTtrig = 700
process.dtTPmonitor.defaultTmax = 200
process.dtTPmonitor.inTimeHitsLowerBound = 0
process.dtTPmonitor.inTimeHitsUpperBound = 0

wiresToDebug = cms.untracked.vstring()
if filename:
    for line in open(filename):
        corrWire = line.split()[:6]
        # switch station/sector
        corrWire[1:3] = corrWire[2:0:-1]
        wire = ' '.join(corrWire)
        # append to wiresToDebug
        wiresToDebug.append(wire)

process.t0calib = cms.EDAnalyzer("DTT0Calibration",
    # Cells for which you want the histos (default = None)
    cellsWithHisto = wiresToDebug,
    # Label to retrieve DT digis from the event
    digiLabel = cms.untracked.string('dtunpacker'),
    calibSector = cms.untracked.string('All'),
    # Chose the wheel, sector (default = All)
    calibWheel = cms.untracked.string('All'),
    # Number of events to be used for the t0 per layer histos
    eventsForWireT0 = cms.uint32(300000),
    # Name of the ROOT file which will contain the test pulse times per layer
    rootFileName = cms.untracked.string('/afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/RUNPERIODTEMPL/t0/DTTestPulses_RUNNUMBERTEMPLATE.root'),
    debug = cms.untracked.bool(False),
    rejectDigiFromPeak = cms.uint32(50),
    # Acceptance for TP peak width
    tpPeakWidth = cms.double(15.0),
    # Number of events to be used for the t0 per layer histos
    eventsForLayerT0 = cms.uint32(50000)
)

process.FEVT = cms.OutputModule("PoolOutputModule",
               outputCommands = cms.untracked.vstring('drop *', 
                                'keep *_MEtoEDMConverter_*_*'),
               fileName = cms.untracked.string('DQM.root')
                               )
process.load("DQMServices.Components.MEtoEDMConverter_cff")
process.dummyProducer = cms.EDProducer("ThingWithMergeProducer")
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.dqmSaver.convention = "Online"
# /cms/mon/data/dropbox
process.dqmSaver.dirName = "."
#process.dqmSaver.dirName = "/tmp/antoniov"
process.dqmSaver.producer = "DQM"
process.dqmSaver.saveByRun         =  1
process.dqmSaver.saveAtJobEnd      = True

process.firstStep = cms.Sequence(process.dummyProducer + process.dtunpacker*process.dtTPmonitor*process.t0calib + process.dqmSaver)

process.p = cms.Path(process.firstStep)
process.outpath = cms.EndPath(process.FEVT)
process.DQM.collectorHost = ''
