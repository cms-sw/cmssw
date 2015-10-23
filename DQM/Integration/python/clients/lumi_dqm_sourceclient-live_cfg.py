import FWCore.ParameterSet.Config as cms

process = cms.Process("DQMLumi")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

#----------------------------
# Event Source
#----------------------------
process.load("DQM.Integration.config.inputsource_cfi")
#process.DQMEventStreamHttpReader.consumerName = 'DQM Luminosity Consumer'
#process.DQMEventStreamHttpReader.SelectHLTOutput = cms.untracked.string('hltOutputALCALUMIPIXELS')

#----------------------------
# DQM Environment
#----------------------------
process.load("DQM.Integration.config.environment_cfi")
process.dqmEnv.subSystemFolder    = "Info/Lumi"
process.dqmSaver.tag = "Lumi"

#---------------------------------------------
# Global Tag
#---------------------------------------------

#--------------------------
#  Lumi Producer and DB access
#-------------------------
process.DBService=cms.Service('DBService',
                              authPath= cms.untracked.string('/nfshome0/popcondev/conddb')
                              )
process.expressLumiProducer=cms.EDProducer("ExpressLumiProducer",
                      connect=cms.string('oracle://cms_omds_lb/CMS_RUNTIME_LOGGER')
                                          )
#---------------------------
#----------------------------
# Sub-system Configuration
#----------------------------
### @@@@@@ Comment when running locally @@@@@@ ###
process.load("DQM.Integration.config.FrontierCondition_GT_cfi")
process.GlobalTag.DBParameters.authenticationPath  = process.DBService.authPath
### @@@@@@ Comment when running locally @@@@@@ ###
process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff")
process.load("Configuration.StandardSequences.RawToDigi_Data_cff")
process.load("Configuration.StandardSequences.EndOfProcess_cff")
process.load("Configuration.EventContent.EventContent_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.siPixelDigis.InputLabel = cms.InputTag("hltFEDSelectorLumiPixels")

process.reconstruction_step = cms.Sequence(
    process.siPixelDigis +
    process.siPixelClusters
)


#---------------------------
# Lumi Monitor
#---------------------------
process.load("DQMServices/Components/DQMLumiMonitor_cfi")
#----------------------------
# Define Sequence
#----------------------------
process.dqmmodules = cms.Sequence(process.dqmEnv
                                  + process.expressLumiProducer
                                  + process.dqmLumiMonitor    
                                  + process.dqmSaver)
#----------------------------
# Proton-Proton Running Stuff
#----------------------------
process.p = cms.Path(process.reconstruction_step *
                     process.dqmmodules)


### process customizations included here
from DQM.Integration.config.online_customizations_cfi import *
process = customise(process)
