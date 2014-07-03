import FWCore.ParameterSet.Config as cms

process = cms.Process("EvFDQM")

#----------------------------
#### Event Source
#----------------------------
process.load("DQM.Integration.test.inputsource_cfi")
process.EventStreamHttpReader.consumerName = 'FEDTest DQM Consumer'
#process.EventStreamHttpReader.maxEventRequestRate = cms.untracked.double(50.0)



#----------------------------
#### DQM Environment
#----------------------------
process.load("DQMServices.Core.DQM_cfg")
#process.DQMStore.referenceFileName = "DT_reference.root"


#----------------------------
#### DQM Environment
#----------------------------
process.load("DQM.Integration.test.environment_cfi")
process.dqmEnv.subSystemFolder = 'FEDTest'
#-----------------------------


# message logger
process.MessageLogger = cms.Service("MessageLogger",
                                    destinations = cms.untracked.vstring('cout'),
                                    cout = cms.untracked.PSet(threshold = cms.untracked.string('WARNING'))
                                    )

# Global tag
process.load("DQM.Integration.test.FrontierCondition_GT_cfi")


#-----------------------------
#### Sub-system configuration follows

# CSC DQM sequence
process.load("DQM.CSCMonitorModule.csc_hlt_dqm_sourceclient_cfi")
process.cscDQMEvF.EventProcessor.FOLDER_EMU = cms.untracked.string('CSC/FEDIntegrity_SM/')

# DT DQM sequence
process.load("DQM.DTMonitorModule.dtDataIntegrityTask_EvF_cff")
process.DTDataIntegrityTask.processingMode = "SM"
process.dtunpacker.inputLabel = cms.InputTag('source')
process.dtunpacker.fedbyType = cms.bool(True)
process.dtunpacker.useStandardFEDid = cms.bool(True)
process.dtunpacker.dqmOnly = cms.bool(True)

# ECAL DQM sequences
process.load("EventFilter.EcalRawToDigiDev.EcalUnpackerMapping_cfi")
process.load("EventFilter.EcalRawToDigiDev.EcalUnpackerData_cfi")
process.load("Geometry.EcalMapping.EcalMapping_cfi")
process.load("Geometry.EcalMapping.EcalMappingRecord_cfi")
import DQM.EcalBarrelMonitorTasks.EBHltTask_cfi
process.ebDQMEvF = DQM.EcalBarrelMonitorTasks.EBHltTask_cfi.ecalBarrelHltTask.clone()
process.ebDQMEvF.folderName = cms.untracked.string('FEDIntegrity_SM')
import DQM.EcalEndcapMonitorTasks.EEHltTask_cfi
process.eeDQMEvF = DQM.EcalEndcapMonitorTasks.EEHltTask_cfi.ecalEndcapHltTask.clone()
process.eeDQMEvF.folderName = cms.untracked.string('FEDIntegrity_SM')

# L1T DQM sequences 
process.load("DQM.L1TMonitor.L1TFED_cfi")
process.l1tfed.FEDDirName=cms.untracked.string("L1T/FEDIntegrity_SM")

# Pixel DQM sequences
process.load("Geometry.TrackerSimData.trackerSimGeometryXML_cfi")
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
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
 			      process.dtunpacker +
 			      process.ecalEBunpacker  + process.ebDQMEvF + process.eeDQMEvF +
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
