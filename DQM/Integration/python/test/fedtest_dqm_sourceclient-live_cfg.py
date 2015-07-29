import FWCore.ParameterSet.Config as cms

process = cms.Process("EvFDQM")

#----------------------------
#### Event Source
#----------------------------
# for live online DQM in P5
process.load("DQM.Integration.config.inputsource_cfi")

# for testing in lxplus
#process.load("DQM.Integration.config.fileinputsource_cfi")

#----------------------------
#### DQM Environment
#----------------------------
process.load("DQMServices.Core.DQM_cfg")

#----------------------------
#### DQM Environment
#----------------------------
process.load("DQM.Integration.config.environment_cfi")
process.dqmEnv.subSystemFolder = 'FEDTest'
process.dqmSaver.tag = 'FEDTest'
# process.dqmSaver.path = '.'
#-----------------------------


# message logger
process.MessageLogger = cms.Service("MessageLogger",
                                    destinations = cms.untracked.vstring('cout'),
                                    cout = cms.untracked.PSet(threshold = cms.untracked.string('WARNING'))
                                    )

# Global tag
process.load("DQM.Integration.config.FrontierCondition_GT_cfi")

# Need for test in lxplus
#process.load("DQM.Integration.config.FrontierCondition_GT_Offline_cfi")


#-----------------------------
#### Sub-system configuration follows

# CSC DQM sequence
process.load("DQM.CSCMonitorModule.csc_hlt_dqm_sourceclient_cfi")
process.cscDQMEvF.EventProcessor.FOLDER_EMU = cms.untracked.string('CSC/FEDIntegrity_SM/')

# DT DQM sequence
process.load("DQM.DTMonitorModule.dtDataIntegrityTask_EvF_cff")
process.DTDataIntegrityTask.processingMode = "SM"
process.DTDataIntegrityTask.fedIntegrityFolder = "DT/FEDIntegrity_SM"
process.dtunpacker.inputLabel = cms.InputTag('source')
process.dtunpacker.fedbyType = cms.bool(True)
process.dtunpacker.useStandardFEDid = cms.bool(True)
process.dtunpacker.dqmOnly = cms.bool(True)

# ECAL DQM sequences
process.load("Geometry.EcalMapping.EcalMapping_cfi")
process.load("Geometry.EcalMapping.EcalMappingRecord_cfi")
from EventFilter.EcalRawToDigi.EcalUnpackerData_cfi import ecalEBunpacker
process.ecalDigis = ecalEBunpacker.clone()
process.load("DQM.EcalMonitorTasks.EcalFEDMonitor_cfi")
process.ecalFEDMonitor.folderName = cms.untracked.string('FEDIntegrity_SM')

# L1T DQM sequences 
process.load("DQM.L1TMonitor.L1TFED_cfi")
process.l1tfed.FEDDirName=cms.untracked.string("L1T/FEDIntegrity_SM")

# Pixel DQM sequences
process.load("Configuration.StandardSequences.MagneticField_cff")
# Pixel RawToDigi conversion
process.load("EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi")
process.siPixelDigis.InputLabel = "source"
process.siPixelDigis.Timing = False
process.siPixelDigis.IncludeErrors = True
process.load("DQM.SiPixelMonitorRawData.SiPixelMonitorHLT_cfi")
process.SiPixelHLTSource.saveFile = False
process.SiPixelHLTSource.slowDown = False
process.SiPixelHLTSource.DirName = cms.untracked.string('Pixel/FEDIntegrity_SM/')

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

# SiStrip DQM sequences
process.load("DQM.SiStripMonitorHardware.siStripFEDCheck_cfi")
process.siStripFEDCheck.DirName = cms.untracked.string('SiStrip/FEDIntegrity_SM/')


# Hcal DQM sequences
process.load("EventFilter.HcalRawToDigi.HcalRawToDigi_cfi")
process.load("DQM.HcalMonitorTasks.HcalDataIntegrityTask_cfi")
process.hcalDataIntegrityMonitor.TaskFolder="FEDIntegrity_SM"

# RPC
process.load("EventFilter.RPCRawToDigi.rpcUnpacker_cfi")
process.load("DQM.RPCMonitorClient.RPCFEDIntegrity_cfi")
process.rpcFEDIntegrity.RPCPrefixDir = cms.untracked.string('RPC/FEDIntegrity_SM')

# ES raw2digi
process.load("EventFilter.ESRawToDigi.esRawToDigi_cfi")
process.load("DQM.EcalPreshowerMonitorModule.ESFEDIntegrityTask_cfi")
process.ecalPreshowerFEDIntegrityTask.FEDDirName=cms.untracked.string("FEDIntegrity_SM")

# FED Integrity Client
process.load("DQMServices.Components.DQMFEDIntegrityClient_cff")
process.dqmFEDIntegrity.moduleName = "FEDTest"
process.dqmFEDIntegrity.fedFolderName = cms.untracked.string("FEDIntegrity_SM")

# DQM Modules
process.dqmmodules = cms.Sequence(process.dqmEnv + process.dqmSaver)
#process.physicsEventsFilter = cms.EDFilter("HLTTriggerTypeFilter",
#                                  # 1=Physics, 2=Calibration, 3=Random, 4=Technical
#                                  SelectedTriggerType = cms.int32(1)
#                                  ) 
#-----------------------------
process.evfDQMPath = cms.Path(#process.physicsEventsFilter+
                              process.cscDQMEvF +
 			      process.dtunpacker + process.DTDataIntegrityTask +
 			      process.ecalDigis  + process.ecalFEDMonitor + 
			      process.l1tfed +
 			      process.siPixelDigis + process.SiPixelHLTSource +
                              process.siStripFEDCheck + 
			      process.hcalDigis + process.hcalDataIntegrityMonitor +
			      process.rpcunpacker + process.rpcFEDIntegrity +
			      process.esRawToDigi + process.ecalPreshowerFEDIntegrityTask +
			      process.dqmFEDIntegrityClient 
)
process.evfDQMmodulesPath = cms.Path(
                              process.dqmmodules 
)
process.schedule = cms.Schedule(process.evfDQMPath,process.evfDQMmodulesPath)

#--------------------------------------------------
# Heavy Ion Specific Fed Raw Data Collection Label
#--------------------------------------------------
process.dtunpacker.inputLabel = cms.InputTag("rawDataCollector")
process.ecalDigis.InputLabel = cms.InputTag("rawDataCollector")
process.esRawToDigi.sourceTag = cms.InputTag("rawDataCollector")
process.hcalDigis.InputLabel = cms.InputTag("rawDataCollector")
process.rpcunpacker.InputLabel = cms.InputTag("rawDataCollector")
process.siPixelDigis.InputLabel = cms.InputTag("rawDataCollector")
process.SiPixelHLTSource.RawInput = cms.InputTag("rawDataCollector")
process.cscDQMEvF.InputObjects = cms.untracked.InputTag("rawDataCollector")
process.ecalFEDMonitor.FEDRawDataCollection = cms.InputTag("rawDataCollector")
process.ecalPreshowerFEDIntegrityTask.FEDRawDataCollection = cms.InputTag("rawDataCollector")
process.hcalDataIntegrityMonitor.RawDataLabel = cms.untracked.InputTag("rawDataCollector")
process.l1tfed.rawTag = cms.InputTag("rawDataCollector")
process.siStripFEDCheck.RawDataTag = cms.InputTag("rawDataCollector")


print "Running with run type = ", process.runType.getRunType()

if (process.runType.getRunType() == process.runType.hi_run):
    process.dtunpacker.inputLabel = cms.InputTag("rawDataRepacker")
    process.ecalDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.esRawToDigi.sourceTag = cms.InputTag("rawDataRepacker")
    process.hcalDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.rpcunpacker.InputLabel = cms.InputTag("rawDataRepacker")
    process.siPixelDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.SiPixelHLTSource.RawInput = cms.InputTag("rawDataRepacker")
    process.cscDQMEvF.InputObjects = cms.untracked.InputTag("rawDataRepacker")
    process.ebDQMEvF.FEDRawDataCollection = cms.InputTag("rawDataRepacker")
    process.ecalPreshowerFEDIntegrityTask.FEDRawDataCollection = cms.InputTag("rawDataRepacker")
    process.eeDQMEvF.FEDRawDataCollection = cms.InputTag("rawDataRepacker")
    process.hcalDataIntegrityMonitor.RawDataLabel = cms.untracked.InputTag("rawDataRepacker")
    process.l1tfed.rawTag = cms.InputTag("rawDataRepacker")
    process.siStripFEDCheck.RawDataTag = cms.InputTag("rawDataRepacker")


### process customizations included here
from DQM.Integration.config.online_customizations_cfi import *
process = customise(process)
