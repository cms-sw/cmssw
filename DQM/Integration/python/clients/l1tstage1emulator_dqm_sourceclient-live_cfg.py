from __future__ import print_function
# L1 Emulator DQM sequence
#
#   authors previous versions - see CVS
#
#   V.M. Ghete 2010-10-22 revised version of L1 emulator DQM


import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_cff import Run3
process = cms.Process("L1TEmuDQMlive", Run3)


#----------------------------
# Event Source
#
# for live online DQM in P5
process.load("DQM.Integration.config.inputsource_cfi")
from DQM.Integration.config.inputsource_cfi import options
#
# for testing in lxplus
#process.load("DQM.Integration.config.fileinputsource_cfi")
#from DQM.Integration.config.fileinputsource_cfi import options

#----------------------------
# DQM Environment
#
 
process.load("DQM.Integration.config.environment_cfi")
# for local test
process.dqmEnv.subSystemFolder = 'L1TEMUStage1'
process.dqmSaver.tag = 'L1TEMUStage1'
process.dqmSaver.runNumber = options.runNumber
process.dqmSaverPB.tag = 'L1TEMUStage1'
process.dqmSaverPB.runNumber = options.runNumber

#
# no references needed

#
# Condition for P5 cluster
process.load("DQM.Integration.config.FrontierCondition_GT_cfi")
process.GlobalTag.RefreshEachRun = True
# Condition for lxplus: change and possibly customise the GT
#from Configuration.AlCa.GlobalTag import GlobalTag as gtCustomise
#process.GlobalTag = gtCustomise(process.GlobalTag, 'auto:run3_data', '')

#process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
#-------------------------------------
# sequences needed for L1 emulator DQM
#

# standard unpacking sequence 
process.load("Configuration.StandardSequences.RawToDigi_Data_cff")    

# L1 data - emulator sequences 
process.load("DQM.L1TMonitor.L1TEmulatorMonitor_cff")    
process.load("DQM.L1TMonitorClient.L1TEMUMonitorClient_cff")    
#process.load("L1Trigger.L1TCalorimeter.caloStage1Params_cfi")

#-------------------------------------
# paths & schedule for L1 emulator DQM
#

# TODO define a L1 trigger L1TriggerRawToDigi in the standard sequence 
# to avoid all these remove
process.rawToDigiPath = cms.Path(process.RawToDigi)
#
process.RawToDigi.remove("siPixelDigis")
process.RawToDigi.remove("siStripDigis")
process.RawToDigi.remove("scalersRawToDigi")
process.RawToDigi.remove("castorDigis")

#if ( process.runType.getRunType() == process.runType.pp_run_stage1 or process.runType.getRunType() == process.runType.cosmic_run_stage1):
process.gtDigis.DaqGtFedId = 813
#else:
#    process.gtDigis.DaqGtFedId = cms.untracked.int32(809)

# L1HvVal + emulator monitoring path
process.l1HwValEmulatorMonitorPath = cms.Path(process.l1Stage1HwValEmulatorMonitor)

# for RCT at P5, read FED vector from OMDS
#process.load("L1TriggerConfig.RCTConfigProducers.l1RCTOmdsFedVectorProducer_cfi")
#process.valRctDigis.getFedsFromOmds = cms.bool(True)

process.stage1UnpackerPath = cms.Path(process.caloStage1Digis+process.caloStage1LegacyFormatDigis)

#
process.l1EmulatorMonitorClientPath = cms.Path(process.l1EmulatorMonitorClient)

#
process.l1EmulatorMonitorEndPath = cms.EndPath(process.dqmEnv*process.dqmSaver*process.dqmSaverPB)

#

#
process.schedule = cms.Schedule(process.rawToDigiPath,
                                process.stage1UnpackerPath,
                                process.l1HwValEmulatorMonitorPath,
                                process.l1EmulatorMonitorClientPath,
                                process.l1EmulatorMonitorEndPath)

#---------------------------------------------

# examples for quick fixes in case of troubles 
#    please do not modify the commented lines
#
# remove a module from hardware validation
# cff file: L1Trigger.HardwareValidation.L1HardwareValidation_cff
#
# process.L1HardwareValidation.remove(process.deCsctf)
#
process.L1HardwareValidation.remove(process.deDt)


#
# remove a L1 trigger system from the comparator integrated in hardware validation
# cfi file: L1Trigger.HardwareValidation.L1Comparator_cfi
#
#process.l1compare.COMPARE_COLLS = [0, 0, 1, 1,  0, 1, 0, 0, 1, 0, 1, 0]
#

#process.l1compareforstage1.COMPARE_COLLS = [
#        0,  0,  0,  1,   0,  0,  0,  0,  0,  0,  0, 0
#        ]

#process.l1demonstage1.COMPARE_COLLS = [
#        0,  0,  0,  1,   0,  0,  0,  0,  0,  0,  0, 0
#        ]
      #ETP,HTP,RCT,GCT, DTP,DTF,CTP,CTF,RPC,LTC,GMT,GT


#
# remove an expert module for L1 trigger system
# cff file: DQM.L1TMonitor.L1TEmulatorMonitor_cff
#
# process.l1ExpertDataVsEmulator.remove(process.l1GtHwValidation)
#

#process.l1ExpertDataVsEmulatorStage1.remove(process.l1TdeCSCTF)

#process.l1ExpertDataVsEmulatorStage1.remove(process.l1TdeRCT)

process.l1demonstage1.HistFolder = 'L1TEMU'

process.l1TdeStage1Layer2.HistFolder = 'L1TEMU/Stage1Layer2expert'

process.l1Stage1GtHwValidation.DirName = "L1TEMU/GTexpert"

#
# remove a module / sequence from l1EmulatorMonitorClient
# cff file: DQM.L1TMonitorClient.L1TEmulatorMonitorClient_cff
#
# process.l1EmulatorMonitorClient.remove(process.l1EmulatorErrorFlagClient)
#


#
# fast over-mask a system in L1TEMUEventInfoClient: 
#   if the name of the system is in the list, the system will be masked
#   (the default mask value is given in L1Systems VPSet)             
#
# names are case sensitive, order is irrelevant
# "ECAL", "HCAL", "RCT", "GCT", "DTTF", "DTTPG", "CSCTF", "CSCTPG", "RPC", "GMT", "GT"
#
# process.l1temuEventInfoClient.DisableL1Systems = cms.vstring("ECAL")
#


#
# fast over-mask an object in L1TEMUEventInfoClient:
#   if the name of the object is in the list, the object will be masked
#   (the default mask value is given in L1Objects VPSet)             
#
# names are case sensitive, order is irrelevant
# 
# "Mu", "NoIsoEG", "IsoEG", "CenJet", "ForJet", "TauJet", "ETM", "ETT", "HTT", "HTM", 
# "HfBitCounts", "HfRingEtSums", "TechTrig", "GtExternal
#
# process.l1temuEventInfoClient.DisableL1Objects =  cms.vstring("ETM")   
#


#
# turn on verbosity in L1TEMUEventInfoClient
#
# process.l1EmulatorEventInfoClient.verbose = cms.untracked.bool(True)

print("Running with run type = ", process.runType.getRunType())
process.castorDigis.InputLabel = "rawDataCollector"
process.csctfDigis.producer = "rawDataCollector"
process.dttfDigis.DTTF_FED_Source = "rawDataCollector"
process.ecalDigisCPU.InputLabel = "rawDataCollector"
process.ecalPreshowerDigis.sourceTag = "rawDataCollector"
process.gctDigis.inputLabel = "rawDataCollector"
process.gtDigis.DaqGtInputTag = "rawDataCollector"
process.gtEvmDigis.EvmGtInputTag = "rawDataCollector"
process.hcalDigis.InputLabel = "rawDataCollector"
process.l1compare.FEDsourceEmul = "rawDataCollector"
process.l1compare.FEDsourceData = "rawDataCollector"
process.muonCSCDigis.InputObjects = "rawDataCollector"
process.muonDTDigis.inputLabel = "rawDataCollector"
process.muonRPCDigis.InputLabel = "rawDataCollector"
process.scalersRawToDigi.scalersInputTag = "rawDataCollector"
process.siPixelDigis.cpu.InputLabel = "rawDataCollector"
process.siStripDigis.ProductLabel = "rawDataCollector"

#--------------------------------------------------
# Heavy Ion Specific Fed Raw Data Collection Label
#--------------------------------------------------
if (process.runType.getRunType() == process.runType.hi_run):
    process.castorDigis.InputLabel = "rawDataRepacker"
    process.csctfDigis.producer = "rawDataRepacker"
    process.dttfDigis.DTTF_FED_Source = "rawDataRepacker"
    process.ecalDigisCPU.InputLabel = "rawDataRepacker"
    process.ecalPreshowerDigis.sourceTag = "rawDataRepacker"
    process.gctDigis.inputLabel = "rawDataRepacker"
    process.gtDigis.DaqGtInputTag = "rawDataRepacker"
    process.gtEvmDigis.EvmGtInputTag = "rawDataRepacker"
    process.hcalDigis.InputLabel = "rawDataRepacker"
    process.l1compare.FEDsourceEmul = "rawDataRepacker"
    process.l1compare.FEDsourceData = "rawDataRepacker"
    process.muonCSCDigis.InputObjects = "rawDataRepacker"
    process.muonDTDigis.inputLabel = "rawDataRepacker"
    process.muonRPCDigis.InputLabel = "rawDataRepacker"
    process.scalersRawToDigi.scalersInputTag = "rawDataRepacker"
    process.siPixelDigis.cpu.InputLabel = "rawDataRepacker"
    process.siStripDigis.ProductLabel = "rawDataRepacker"



### process customizations included here
from DQM.Integration.config.online_customizations_cfi import *
process = customise(process)
