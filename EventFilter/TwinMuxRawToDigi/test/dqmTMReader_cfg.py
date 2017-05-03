
import FWCore.ParameterSet.Config as cms

process = cms.Process("DTDQM")

#----------------------------
#### Event Source
#----------------------------
process.load("DQM.DTMonitorModule.test.NewEventStreamFileReader_cfi")

#----------------------------
#### DQM Environment
#----------------------------
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")

##----------------------------
##### DQM Live Environment
##----------------------------
process.dqmEnv.subSystemFolder  = 'TM'
process.dqmSaver.convention     = 'Online'
process.dqmSaver.dirName        = '.'
process.dqmSaver.producer       = 'DQM'

#process.dqmSaver.saveByTime         = -1
process.dqmSaver.saveByLumiSection  = -1
#process.dqmSaver.saveByMinute       = -1
process.dqmSaver.saveByRun          = 1
process.dqmSaver.saveAtJobEnd       = True
#-----------------------------

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')

# DT reco and DQM sequences
process.load("DQM.DTMonitorModule.dt_dqm_sourceclient_common_cff")
#---- for P5 (online) DB access
#process.GlobalTag.connect ="frontier://(proxyurl=http://localhost:3128)(serverurl=http://frontier1.cms:8000/FrontierOnProd)(serverurl=http://frontier2.cms:8000/FrontierOnProd)(retrieve-ziplevel=0)/CMS_COND_21X_GLOBALTAG"
#process.GlobalTag.globaltag = cms.string( "74X_dataRun2_Prompt_v1::All" )
process.GlobalTag.globaltag = cms.string( "GR_E_V49::All" )
#---- for offline DB
#process.GlobalTag.globaltag = "CRAFT_V2P::All"

# message logger
process.MessageLogger = cms.Service("MessageLogger",
                                    destinations = cms.untracked.vstring('cout'),
                                    cout = cms.untracked.PSet(threshold = cms.untracked.string('WARNING'))
                                    )

#########################
###our changes
process.load("EventFilter.TwinMuxRawToDigi.dttmunpacker_cfi")
process.unpackers = cms.Sequence(process.dtunpacker + process.dttm7unpacker + process.scalersRawToDigi)
# substitutions
process.dtTPTriggerMonitor.dcc_label = cms.untracked.string('dttm7unpacker')
process.dtTriggerBaseMonitor.inputTagDCC = cms.untracked.InputTag("dttm7unpacker")
process.dtTriggerBaseMonitor.detailedAnalysis = cms.untracked.bool(True)
process.dtTriggerBaseMonitor.minBXDCC = cms.untracked.int32(0)
process.dtTriggerBaseMonitor.maxBXDCC = cms.untracked.int32(2)
process.dtTriggerLutMonitor.inputTagDCC = cms.untracked.InputTag("dttm7unpacker")
# sequence of DQM tasks to be run on physics events only
process.dtDQMTask = cms.Sequence(process.dtTriggerBaseMonitor)
# DQM clients to be run on physics event only
process.dtDQMTest = cms.Sequence(process.triggerTest)
#########################

process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('drop *', 
        'keep L1MuDTChambPhContainer_*_*_*', 
        'keep L1MuDTChambThContainer_*_*_*'),
    fileName = cms.untracked.string('DTTriggerPrimitivesTM.root')
)

process.this_is_the_end = cms.EndPath(process.out)

### back to dqm
process.dqmmodules = cms.Sequence(process.dqmEnv + process.dqmSaver)

process.dtDQMPathPhys = cms.Path(process.unpackers + process.dqmmodules + process.reco + process.dtDQMTask + process.dtDQMTest)

############ DTTM
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10000) )
#process.MessageLogger.cerr.FwkReport.reportEvery = 1000

process.source = cms.Source("PoolSource",                 
                            fileNames = cms.untracked.vstring(  
                            '/store/data/Run2015D/SingleMuon/RAW/v1/000/258/320/00000/000EF727-DF6B-E511-9763-02163E0146D5.root',
                            )                            
                            
)
