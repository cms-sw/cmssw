import FWCore.ParameterSet.Config as cms
process = cms.Process("test")

### event source
process.source = cms.Source("EmptySource")

### set number of events
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10000)
    )

### include to get DQM histogramming services
process.load("DQMServices.Core.DQM_cfg")

### include your reference file
#process.DQMStore.referenceFileName = 'ref.root'

### set the verbose
process.DQMStore.verbose = 0

###  DQM Source program (in DQMServices/Examples/src/DQMSourceExample.cc)
process.dqmSource   = cms.EDFilter("DQMSourceExample",
        monitorName = cms.untracked.string('YourSubsystemName'),
        prescaleEvt = cms.untracked.int32(1),
        prescaleLS  =  cms.untracked.int32(1)                    
)

### run the quality tests as defined in QualityTests.xml
process.qTester = cms.EDFilter("QualityTester",
    qtList          = cms.untracked.FileInPath('DQMServices/Examples/test/QualityTests.xml'),
    prescaleFactor  = cms.untracked.int32(1),                               
    testInEventloop = cms.untracked.bool(True),
    verboseQT       =  cms.untracked.bool(True)                 
)

###  DQM Client program (in DQMServices/Examples/src/DQMClientExample.cc)
process.dqmClient = cms.EDFilter("DQMClientExample",
    monitorName   = cms.untracked.string('YourSubsystemName'),
    QTestName     = cms.untracked.string('YRange'),                     
    prescaleLS    = cms.untracked.int32(1),  
    prescaleEvt   = cms.untracked.int32(1)
)

# MessageLogger
process.MessageLogger = cms.Service("MessageLogger",
                #suppressWarning = cms.untracked.vstring('qTester')      
               #destinations = cms.untracked.vstring('detailedInfo'),
               #debugModules = cms.untracked.vstring('*'),
               #detailedInfo = cms.untracked.PSet(
               #threshold = cms.untracked.string('DEBUG')
               #                )
)

#### BEGIN DQM Online Environment #######################
    
### replace YourSubsystemName by the name of your source ###
### use it for dqmEnv, dqmSaver
process.load("DQMServices.Components.DQMEnvironment_cfi")
#process.DQM.collectorHost = 'srv-c2d05-XX'
#process.DQM.collectorPort = 9190

### path where to save the output file
process.dqmSaver.dirName = '.'

### the filename prefix 
process.dqmSaver.producer = 'DQM'

### possible conventions are "Online", "Offline" and "RelVal"
process.dqmSaver.convention = 'Online'

process.dqmEnv.subSystemFolder = 'YourSubsystemName'

### optionally change fileSaving  conditions
#process.dqmSaver.saveByLumiSection = -1
#process.dqmSaver.saveByMinute      = -1
#process.dqmSaver.saveByEvent       = -1
#process.dqmSaver.saveByRun         =  1
#process.dqmSaver.saveAtJobEnd      = False


### FIX YOUR  PATH TO INCLUDE dqmEnv and dqmSaver
process.p = cms.Path(process.dqmSource*process.qTester*process.dqmClient*process.dqmEnv*process.dqmSaver)

