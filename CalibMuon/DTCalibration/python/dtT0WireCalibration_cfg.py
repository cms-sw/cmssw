import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff")
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

process.load("CalibMuon.DTCalibration.dt_offlineAnalysis_common_cff")

"""
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
"""

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

#process.eventInfoProvider = cms.EDFilter("EventCoordinatesSource",
#    eventInfoFolder = cms.untracked.string('EventInfo/')
#)

# Test pulse monitoring
process.load("DQM.DTMonitorModule.dtDigiTask_TP_cfi")
process.load("DQM.DTMonitorClient.dtOccupancyTest_TP_cfi")
process.dtTPmonitor.defaultTtrig = 300
process.dtTPmonitor.defaultTmax = 100
process.dtTPmonitor.inTimeHitsLowerBound = 0
process.dtTPmonitor.inTimeHitsUpperBound = 0

process.load('CalibMuon.DTCalibration.dtT0WireCalibration_cfi')

process.output = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('drop *', 
                                           'keep *_MEtoEDMConverter_*_*'),
    fileName = cms.untracked.string('DQM.root')
)

process.load("DQMServices.Components.MEtoEDMConverter_cff")
#process.DQM.collectorHost = ''

process.p = cms.Path(process.muonDTDigis*
                     process.dtTPmonitor+process.dtTPmonitorTest+
                     process.dtT0WireCalibration+
                     process.MEtoEDMConverter)
process.outpath = cms.EndPath(process.output)
