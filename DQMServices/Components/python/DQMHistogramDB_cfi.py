import FWCore.ParameterSet.Config as cms


dqmHistogramDB = cms.EDAnalyzer("DQMHistogramDB",

    statsDepth = cms.untracked.int32(1),
    pathNameMatch = cms.untracked.string('*'),
    dumpMemoryHistory = cms.untracked.bool(True),                             
    verbose = cms.untracked.int32(0),
    runInEventLoop = cms.untracked.bool(False),
    dumpOnEndLumi = cms.untracked.bool(True),
    dumpOnEndRun = cms.untracked.bool(True),
    runOnEndJob = cms.untracked.bool(False),
    dumpToFWJR = cms.untracked.bool(True),
    histogramNamesEndLumi = cms.untracked.vstring("AlcaBeamMonitor/Debug/hsigmaXCoordinate"),

    histogramNamesEndRun = cms.untracked.vstring("AlcaBeamMonitor/Debug/hsigmaXCoordinate",
						"AlcaBeamMonitor/Debug/hsigmaYCoordinate",
						"AlcaBeamMonitor/Debug/hsigmaZCoordinate",
					   	"AlcaBeamMonitor/Debug/hxCoordinate",
					   	"AlcaBeamMonitor/Debug/hyCoordinate",
						"AlcaBeamMonitor/Debug/hzCoordinate"),
    

    #database configuration
    DBParameters = cms.PSet(authenticationPath = cms.untracked.string(''),
			messageLevel = cms.untracked.int32(3),
			enableConnectionSharing = cms.untracked.bool(True),
			connectionTimeOut = cms.untracked.int32(60),
			enableReadOnlySessionOnUpdateConnection = cms.untracked.bool(False),
			connectionRetrialTimeOut = cms.untracked.int32(60),
			connectionRetrialPeriod = cms.untracked.int32(10),
			enablePoolAutomaticCleanUp = cms.untracked.bool(False)
			),

    connect = cms.string('sqlite_file:db2.db'),
)
