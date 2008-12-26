import FWCore.ParameterSet.Config as cms    

process = cms.Process("CONDOBJMON")
#-------------------------------------------------
# CALIBRATION
#-------------------------------------------------
process.load("DQM.SiStripMonitorSummary.Tags21X_cff")


#-------------------------------------------------
# DQM
#-------------------------------------------------
process.load("DQM.SiStripMonitorSummary.SiStripMonitorCondDataOffline_cfi")

## SI STRIP MONITOR TRACK:
process.load("DQM.SiStripCommon.TkHistoMap_cfi");

process.source = cms.Source("EmptyIOVSource",
    lastRun = cms.untracked.uint32(insert_runnumber),
    timetype = cms.string('runnumber'),
    firstRun = cms.untracked.uint32(insert_runnumber),
    interval = cms.uint32(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('SiStripMonitorCondData'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('Error')
    ),
    destinations = cms.untracked.vstring('error.log', 
        'cout')
)

process.qTester = cms.EDFilter("QualityTester",
    qtList = cms.untracked.FileInPath('DQM/SiStripMonitorSummary/data/insert_QtestsFileName'),
    QualityTestPrescaler = cms.untracked.int32(1),
    getQualityTestsFromFile = cms.untracked.bool(True)
)

process.DQMStore = cms.Service("DQMStore",
    referenceFileName = cms.untracked.string(''),
    verbose = cms.untracked.int32(1)
)



    #-------------------------------------------------
    ## Scheduling
    #-------------------------------------------------

####process.p = cms.Path(process.CondDataMonitoring*process.qTester)
process.p = cms.Path(process.CondDataMonitoring)





  

