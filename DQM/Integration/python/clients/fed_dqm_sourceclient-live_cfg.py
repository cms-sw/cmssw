import FWCore.ParameterSet.Config as cms
import sys

# Process initialization
process = cms.Process('FED')

unitTest = False
if 'unitTest=True' in sys.argv:
    unitTest=True

# Logging:
process.MessageLogger = cms.Service(
    'MessageLogger',
    destinations = cms.untracked.vstring('cout'),
    cout = cms.untracked.PSet(threshold = cms.untracked.string('ERROR'))
                                   )

# Global configuration

# DQM Environment:
process.load('DQMServices.Core.DQM_cfg')
process.load('DQM.Integration.config.environment_cfi')
# Global tag:
process.load('DQM.Integration.config.FrontierCondition_GT_cfi')
# Input:
if unitTest:
    process.load("DQM.Integration.config.unittestinputsource_cfi")
    from DQM.Integration.config.unittestinputsource_cfi import options
else:
    process.load('DQM.Integration.config.inputsource_cfi')
    from DQM.Integration.config.inputsource_cfi import options
# Output:
process.dqmEnv.subSystemFolder = 'FED'
process.dqmSaver.tag = 'FED'
process.dqmSaver.runNumber = options.runNumber
process.dqmSaverPB.tag = 'FED'
process.dqmSaverPB.runNumber = options.runNumber

# Subsystem sequences

# We will reuse the same foldername for all subsystems:
folder_name = 'FEDIntegrity_EvF'

# L1T sequence:
process.load('DQM.L1TMonitor.L1TStage2FED_cff') # stage2 L1T
path = 'L1T/%s/' % folder_name
process.l1tStage2Fed.FEDDirName = cms.untracked.string(path)
# Pixel sequence:
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi')
process.siPixelDigis.cpu.Timing = False
process.siPixelDigis.cpu.IncludeErrors = True
process.load('DQM.SiPixelMonitorRawData.SiPixelMonitorHLT_cfi')
process.SiPixelHLTSource.saveFile = False
process.SiPixelHLTSource.slowDown = False
path = 'Pixel/%s/' % folder_name
process.SiPixelHLTSource.DirName = cms.untracked.string(path)
process.load('Configuration.StandardSequences.GeometryRecoDB_cff') # ???
# SiStrip sequence:
process.load('DQM.SiStripMonitorHardware.siStripFEDCheck_cfi')
path = 'SiStrip/%s/' % folder_name
process.siStripFEDCheck.DirName = cms.untracked.string(path)
# ECAL Preshower sequence:
process.load('EventFilter.ESRawToDigi.esRawToDigi_cfi')
process.load('DQM.EcalPreshowerMonitorModule.ESFEDIntegrityTask_cfi')
process.ecalPreshowerFEDIntegrityTask.FEDDirName = cms.untracked.string(folder_name)
# ECAL sequence --> Both ECAL Barrel and ECAL Endcap:
process.load('Geometry.EcalMapping.EcalMapping_cfi')
process.load('Geometry.EcalMapping.EcalMappingRecord_cfi')
from EventFilter.EcalRawToDigi.EcalUnpackerData_cfi import ecalEBunpacker
process.ecalDigis = ecalEBunpacker.clone()
process.load('DQM.EcalMonitorTasks.EcalFEDMonitor_cfi')
process.ecalFEDMonitor.folderName = cms.untracked.string(folder_name)
# HCAL sequence:
process.load('EventFilter.HcalRawToDigi.HcalRawToDigi_cfi')
# DT sequence:
process.load('DQM.DTMonitorModule.dtDataIntegrityTask_EvF_cff')
process.dtDataIntegrityTask.processingMode = 'SM'
path = 'DT/%s/' % folder_name
process.dtDataIntegrityTask.fedIntegrityFolder = path
process.dtDataIntegrityTask.dtFEDlabel     = 'dtunpacker'
# RPC sequence:
process.load('EventFilter.RPCRawToDigi.rpcUnpacker_cfi')
process.load('DQM.RPCMonitorClient.RPCFEDIntegrity_cfi')
path = 'RPC/%s/' % folder_name
process.rpcFEDIntegrity.RPCPrefixDir = cms.untracked.string(path)
# CSC sequence:
process.load('DQM.CSCMonitorModule.csc_hlt_dqm_sourceclient_cfi')
path = 'CSC/%s/' % folder_name
process.cscDQMEvF.EventProcessor.FOLDER_EMU = cms.untracked.string(path)

# Setting raw data collection label for all subsytem modules, depending on run type:
if (process.runType.getRunType() == process.runType.hi_run):
    process.l1tStage2Fed.rawTag = cms.InputTag('rawDataRepacker')
    process.siPixelDigis.cpu.InputLabel = cms.InputTag('rawDataRepacker')
    process.SiPixelHLTSource.RawInput = cms.InputTag('rawDataRepacker')
    process.siStripFEDCheck.RawDataTag = cms.InputTag('rawDataRepacker')
    process.esRawToDigi.sourceTag = cms.InputTag('rawDataRepacker')
    process.ecalPreshowerFEDIntegrityTask.FEDRawDataCollection = cms.InputTag('rawDataRepacker')
    process.ecalDigis.InputLabel = cms.InputTag('rawDataRepacker')
    process.ecalFEDMonitor.FEDRawDataCollection = cms.InputTag('rawDataRepacker')
    process.hcalDigis.InputLabel = cms.InputTag('rawDataRepacker')
    process.dtunpacker.inputLabel = cms.InputTag('rawDataRepacker')
    process.rpcunpacker.InputLabel = cms.InputTag('rawDataRepacker')
    process.cscDQMEvF.InputObjects = cms.untracked.InputTag('rawDataRepacker')
else:
    process.l1tStage2Fed.rawTag = cms.InputTag('rawDataCollector')
    process.siPixelDigis.cpu.InputLabel = cms.InputTag('rawDataCollector')
    process.SiPixelHLTSource.RawInput = cms.InputTag('rawDataCollector')
    process.siStripFEDCheck.RawDataTag = cms.InputTag('rawDataCollector')
    process.esRawToDigi.sourceTag = cms.InputTag('rawDataCollector')
    process.ecalPreshowerFEDIntegrityTask.FEDRawDataCollection = cms.InputTag('rawDataCollector')
    process.ecalDigis.InputLabel = cms.InputTag('rawDataCollector')
    process.ecalFEDMonitor.FEDRawDataCollection = cms.InputTag('rawDataCollector')
    process.hcalDigis.InputLabel = cms.InputTag('rawDataCollector')
    process.dtunpacker.inputLabel = cms.InputTag('rawDataCollector')
    process.rpcunpacker.InputLabel = cms.InputTag('rawDataCollector')
    process.cscDQMEvF.InputObjects = cms.untracked.InputTag('rawDataCollector')

# Finaly the DQM FED sequence itself
process.load('DQMServices.Components.DQMFEDIntegrityClient_cff')
process.dqmFEDIntegrity.fedFolderName = cms.untracked.string(folder_name)

# Sequences, paths and schedules:

# Modules for the FED
process.FEDModulesPath = cms.Path(
			                        process.l1tStage2Fed
			                      + process.siPixelDigis
                                  + process.SiPixelHLTSource
                                  + process.siStripFEDCheck
			                      + process.esRawToDigi
                                  + process.ecalPreshowerFEDIntegrityTask
 			                      + process.ecalDigis
                                  + process.ecalFEDMonitor
			                      + process.hcalDigis
                                  + process.cscDQMEvF
 			                      + process.dtunpacker
                                  + process.dtDataIntegrityTask
			                      + process.rpcunpacker
                                  + process.rpcFEDIntegrity

			                      + process.dqmFEDIntegrityClient 
                                 )

# Standard DQM modules
process.DQMmodulesPath = cms.Path(
                                    process.dqmEnv
                                  + process.dqmSaver
                                  + process.dqmSaverPB
                                 )

process.schedule = cms.Schedule(
                                 process.FEDModulesPath, 
                                 process.DQMmodulesPath,
                               )

# Finaly: DQM process customizations
from DQM.Integration.config.online_customizations_cfi import *
process = customise(process)
