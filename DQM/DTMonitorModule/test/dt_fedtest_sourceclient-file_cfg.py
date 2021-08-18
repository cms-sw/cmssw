import FWCore.ParameterSet.Config as cms

process = cms.Process("EvFDQM")

#----------------------------
#### Event Source
#----------------------------

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    '/store/data/Commissioning10/Cosmics/RAW/v3/000/130/772/00A80040-A42D-DF11-A17C-000423D990CC.root'
    )
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(5000) )


#----------------------------
#### DQM Environment
#----------------------------
process.load("DQMServices.Core.DQM_cfg")

#----------------------------
#### DQM Environment
#----------------------------
process.load("DQM.Integration.test.environment_cfi")
process.dqmEnv.subSystemFolder = 'FEDTest'
process.dqmSaver.dirName = '.'
process.dqmSaver.saveByRun = True
#-----------------------------

# message logger
process.MessageLogger = cms.Service("MessageLogger",
                                    destinations = cms.untracked.vstring('cout'),
                                    cout = cms.untracked.PSet(threshold = cms.untracked.string('WARNING'))
                                    )

# Global tag
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cfi")
process.GlobalTag.globaltag ="GR10_P_V4::All"

#-----------------------------
#### Sub-system configuration follows

# DT DQM stuff
process.load("DQM.DTMonitorModule.dtDataIntegrityTask_EvF_cfi")

process.load("EventFilter.DTRawToDigi.dtunpacker_cfi")
# enable DQM monitoring in unpacker
process.dtunpacker = process.muonDTDigis.clone(
   performDataIntegrityMonitor = True,
   readOutParameters.performDataIntegrityMonitor = True
)
# DQM Modules
process.dqmmodules = cms.Sequence(process.dqmEnv + process.dqmSaver)

### Define the path
process.dtUnpackAndDQM = cms.Path( process.dtunpacker ) #has to be run in hlt paths nothing has to be added to DQM-hlt paths
process.evfDQMmodulesPath = cms.Path( process.dqmmodules ) 
process.schedule = cms.Schedule(process.dtUnpackAndDQM,process.evfDQMmodulesPath)
