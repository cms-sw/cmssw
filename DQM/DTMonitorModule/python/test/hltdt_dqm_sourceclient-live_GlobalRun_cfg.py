import FWCore.ParameterSet.Config as cms

process = cms.Process("HLT-DTDQM")

#----------------------------
#### Event Source
#----------------------------
process.load("DQM.Integration.test.inputsource_cfi")
process.EventStreamHttpReader.consumerName = 'DT DQM Consumer'


#----------------------------
#### DQM Environment
#----------------------------
process.load("DQMServices.Core.DQM_cfg")
#process.DQMStore.referenceFileName = "DT_reference.root"

process.load("DQMServices.Components.DQMEnvironment_cfi")


#----------------------------
#### DQM Live Environment
#----------------------------
process.load("DQM.Integration.test.environment_cfi")
process.DQM.collectorHost = 'srv-c2d05-13.cms'
process.DQM.collectorPort = 9190
process.dqmEnv.subSystemFolder = 'DT'
process.dqmSaver.convention = 'Online'
process.dqmSaver.dirName = '/localdatadisk/DTDQM/dqmdata'
process.dqmSaver.producer = 'DQM'

process.dqmSaver.saveByTime = -1
process.dqmSaver.saveByLumiSection = -1
process.dqmSaver.saveByMinute = -1
process.dqmSaver.saveByRun = 1
process.dqmSaver.saveAtJobEnd = True
#-----------------------------


# DT reco and DQM sequences
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

# filter on trigger type
physicsEventsFilter = cms.EDFilter("HLTTriggerTypeFilter",
                                   # 1=Physics, 2=Calibration, 3=Random, 4=Technical
                                   SelectedTriggerType = cms.int32(1) 
                                   )


from Configuration.StandardSequences.FrontierConditions_GlobalTag_cff import *
#---- for P5 (online) DB access
process.GlobalTag.connect ="frontier://(proxyurl=http://localhost:3128)(serverurl=http://frontier1.cms:8000/FrontierOnProd)(serverurl=http://frontier2.cms:8000/FrontierOnProd)(retrieve-ziplevel=0)/CMS_COND_31X_GLOBALTAG"
process.GlobalTag.globaltag = "CRAFT_V17H::All"
#---- for offline DB
#process.GlobalTag.globaltag = "CRAFT_V2P::All"

# segment reco task
process.load("DQM.DTMonitorModule.dtSegmentTask_cfi")
process.dtSegmentAnalysisMonitor.recHits4DLabel = 'hltDt4DSegments'
process.dtSegmentAnalysisMonitor.topHistoFolder = "HLT/DTSegments"
process.dtSegmentAnalysisMonitor.hltDQMMode = True
process.load("DQM.DTMonitorClient.dtSegmentAnalysisTest_cfi")
process.segmentTest.topHistoFolder = "HLT/DTSegments"
process.segmentTest.hltDQMMode = True


# resolution task
process.load("DQM.DTMonitorModule.dtResolutionTask_cfi")
process.dtResolutionAnalysisMonitor.recHits4DLabel = 'hltDt4DSegments'
process.dtResolutionAnalysisMonitor.topHistoFolder = "HLT/DTSegments"
process.load("DQM.DTMonitorClient.dtResolutionAnalysisTest_cfi")
process.dtResolutionAnalysisTest.topHistoFolder = "HLT/DTSegments"


# message logger
process.MessageLogger = cms.Service("MessageLogger",
                                    destinations = cms.untracked.vstring('cout'),
                                    cout = cms.untracked.PSet(threshold = cms.untracked.string('WARNING'))
                                    )

process.dqmmodules = cms.Sequence(process.dqmEnv + process.dqmSaver)

process.dtDQMTask = cms.Sequence(dtSegmentAnalysisMonitor + dtResolutionAnalysisMonitor)

process.dtDQMTest = cms.Sequence(segmentTest + dtResolutionAnalysisTest)

process.dtDQMPathPhys = cms.Path(process.dqmmodules + process.physicsEventsFilter * process.dtDQMTask + process.dtDQMTest)



