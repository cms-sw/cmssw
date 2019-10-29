import FWCore.ParameterSet.Config as cms
process = cms.Process("test")

### event source
process.source = cms.Source("EmptySource")

### set number of events
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
    )

######################################################################################
### include to get DQM histogramming services
process.load("DQMServices.Core.DQM_cfg")
process.DQMStore.verbose = 2

### include to get DQM environment (file saver and eventinfo module)
process.load("DQMServices.Components.DQMEnvironment_cfi")

### replace YourSubsystemName by the name of your source ###
### use it for dqmEnv, dqmSaver ###
process.dqmEnv.subSystemFolder = 'YourSubsystemName'

### optional parameters (defaults are different) ###
### Online environment
process.dqmSaver.convention = 'Online'
process.DQM.collectorHost = ''
process.DQM.collectorPort = 9190

### path where to save the output file
### optionally change fileSaving conditions
process.dqmSaver.dirName = '.'
process.dqmSaver.saveByTime = 4
process.dqmSaver.saveByLumiSection = -1
process.dqmSaver.saveByMinute = 8
process.dqmSaver.saveByRun = 1
process.dqmSaver.saveAtJobEnd = True

######################################################################################
### include your reference file
process.DQMStore.referenceFileName = 'ref.root'

######################################################################################
### set this in order to add up histograms that already exist
#process.DQMStore.collateHistograms = cms.untracked.bool(True)

######################################################################################
### loading of root files into DQMStore (stripping out Run and RunSummary)
process.load("DQMServices.Components.DQMFileReader_cfi")
process.dqmFileReader.FileNames = cms.untracked.vstring ( 
       "file:ref.root",
       "file:ref.root",
       "file:ref.root"
       )

######################################################################################
###  DQM Source program (in DQMServices/Examples/src/DQMSourceExample.cc)
process.dqmSource   = cms.EDAnalyzer("DQMSourceExample",
        monitorName = cms.untracked.string('YourSubsystemName'),
        prescaleEvt = cms.untracked.int32(1),
        prescaleLS  =  cms.untracked.int32(1)                    
)

######################################################################################
### run the quality tests as defined in QualityTests.xml
### by default: the quality tests run at the end of each lumisection
from DQMServices.Core.DQMQualityTester import DQMQualityTester
process.qTester    = DQMQualityTester(
    qtList = cms.untracked.FileInPath('DQMServices/Examples/test/QualityTests.xml'),
    prescaleFactor = cms.untracked.int32(1),                               
    testInEventloop = cms.untracked.bool(False), #run on each event
    verboseQT =  cms.untracked.bool(True)                 
)

######################################################################################
### include to get DQM histogramming services
process.load("DQMServices.Components.DQMStoreStats_cfi")

######################################################################################
### DQM Client program (in DQMServices/Examples/src/DQMClientExample.cc)
### by default: the client runs at the end of each lumisection
process.dqmClient = cms.EDAnalyzer("DQMClientExample",
    monitorName   = cms.untracked.string('YourSubsystemName'),
    QTestName     = cms.untracked.string('YRange'),                     
    prescaleEvt   = cms.untracked.int32(1),
    prescaleLS    =  cms.untracked.int32(1),                   
    clientOnEachEvent = cms.untracked.bool(False) #run client on each event
)

######################################################################################
### MessageLogger
process.MessageLogger = cms.Service("MessageLogger",
               #suppressWarning = cms.untracked.vstring('qTester')      
               destinations = cms.untracked.vstring('detailedInfo'),
               #debugModules = cms.untracked.vstring('*'),
               #detailedInfo = cms.untracked.PSet(
               #threshold = cms.untracked.string('DEBUG')
               #                )
)

######################################################################################
### LogError Histogramming
process.load("FWCore.Modules.logErrorHarvester_cfi")
process.load("DQMServices.Components.DQMMessageLogger_cfi")


######################################################################################
process.p = cms.Path(
                     process.dqmFileReader*
		     process.dqmEnv*
                     process.dqmSource*
                     process.qTester*
		     process.dqmStoreStats*
		     process.dqmClient*
#		     process.logErrorHarvester*process.logErrorDQM*
		     process.dqmSaver
		    )
