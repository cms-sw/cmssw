# L1 Emulator DQM sequence
#
#   authors previous versions - see CVS
#
#   V.M. Ghete 2010-10-22 revised version of L1 emulator DQM


import FWCore.ParameterSet.Config as cms

process = cms.Process("L1TEmuDQMlive")


#----------------------------
# Event Source
#
# for live online DQM in P5
process.load("DQM.Integration.config.inputsource_cfi")
#
# for testing in lxplus
#process.load("DQM.Integration.config.fileinputsource_cfi")

#----------------------------
# DQM Environment
#
 

#
process.load("DQM.Integration.config.environment_cfi")
# for local test
process.dqmEnv.subSystemFolder = 'L1TEMU'
process.dqmSaver.tag = 'L1TEMU'
#
# no references needed
# replace DQMStore.referenceFileName = "L1TEMU_reference.root"

#
# Condition for P5 cluster
process.load("DQM.Integration.config.FrontierCondition_GT_cfi")
process.GlobalTag.RefreshEachRun = cms.untracked.bool(True)
# Condition for lxplus
#process.load("DQM.Integration.config.FrontierCondition_GT_Offline_cfi") 

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

# L1HvVal + emulator monitoring path
process.l1HwValEmulatorMonitorPath = cms.Path(process.l1HwValEmulatorMonitor)

#if (process.runType.getRunType() == process.runType.pp_run_stage1 or process.runType.getRunType() == process.runType.cosmic_run_stage1):
process.gtDigis.DaqGtFedId = cms.untracked.int32(809)
#else:
#process.gtDigis.DaqGtFedId = cms.untracked.int32(813)
# for RCT at P5, read FED vector from OMDS
#process.load("L1TriggerConfig.RCTConfigProducers.l1RCTOmdsFedVectorProducer_cfi")
#process.valRctDigis.getFedsFromOmds = cms.bool(True)

#
process.l1EmulatorMonitorClientPath = cms.Path(process.l1EmulatorMonitorClient)

#
process.l1EmulatorMonitorEndPath = cms.EndPath(process.dqmEnv*process.dqmSaver)

#

#
process.schedule = cms.Schedule(process.rawToDigiPath,
                                process.l1HwValEmulatorMonitorPath,
                                #process.l1EmulatorMonitorClientPath,
                                process.l1EmulatorMonitorEndPath)

#---------------------------------------------

# examples for quick fixes in case of troubles 
#    please do not modify the commented lines
#
# remove a module from hardware validation
# cff file: L1Trigger.HardwareValidation.L1HardwareValidation_cff
#
process.l1HwValEmulatorMonitorPath.remove(process.l1TdeCSCTF)
#
process.L1HardwareValidation.remove(process.deDt)

process.L1HardwareValidation.remove(process.l1TdeRCTRun1)
#
# remove a L1 trigger system from the comparator integrated in hardware validation
# cfi file: L1Trigger.HardwareValidation.L1Comparator_cfi
#
# process.l1compare.COMPARE_COLLS = [0, 0, 1, 1,  0, 1, 0, 0, 1, 0, 1, 0]
#
process.l1compare.COMPARE_COLLS = [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]

process.l1demon.COMPARE_COLLS = [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]

process.l1demon.HistFolder = cms.untracked.string('L1TEMU/Legacy')

process.l1TdeGCT.HistFolder = cms.untracked.string('L1TEMU/Legacy/GCTexpert')

process.l1GtHwValidation.DirName = cms.untracked.string("L1TEMU/Legacy/GTexpert")

#
# remove an expert module for L1 trigger system
# cff file: DQM.L1TMonitor.L1TEmulatorMonitor_cff
#
# process.l1ExpertDataVsEmulator.remove(process.l1GtHwValidation)
#

#process.l1ExpertDataVsEmulator.remove(process.l1TdeCSCTF)

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

print "Running with run type = ", process.runType.getRunType()
process.castorDigis.InputLabel = cms.InputTag("rawDataCollector")
process.csctfDigis.producer = cms.InputTag("rawDataCollector")
process.dttfDigis.DTTF_FED_Source = cms.InputTag("rawDataCollector")
process.ecalDigis.InputLabel = cms.InputTag("rawDataCollector")
process.ecalPreshowerDigis.sourceTag = cms.InputTag("rawDataCollector")
process.gctDigis.inputLabel = cms.InputTag("rawDataCollector")
process.gtDigis.DaqGtInputTag = cms.InputTag("rawDataCollector")
process.gtEvmDigis.EvmGtInputTag = cms.InputTag("rawDataCollector")
process.hcalDigis.InputLabel = cms.InputTag("rawDataCollector")
process.l1compare.FEDsourceEmul = cms.untracked.InputTag("rawDataCollector")
process.l1compare.FEDsourceData = cms.untracked.InputTag("rawDataCollector")
process.muonCSCDigis.InputObjects = cms.InputTag("rawDataCollector")
process.muonDTDigis.inputLabel = cms.InputTag("rawDataCollector")
process.muonRPCDigis.InputLabel = cms.InputTag("rawDataCollector")
process.scalersRawToDigi.scalersInputTag = cms.InputTag("rawDataCollector")
process.siPixelDigis.InputLabel = cms.InputTag("rawDataCollector")
process.siStripDigis.ProductLabel = cms.InputTag("rawDataCollector")

#--------------------------------------------------
# Heavy Ion Specific Fed Raw Data Collection Label
#--------------------------------------------------
if (process.runType.getRunType() == process.runType.hi_run):
    process.castorDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.csctfDigis.producer = cms.InputTag("rawDataRepacker")
    process.dttfDigis.DTTF_FED_Source = cms.InputTag("rawDataRepacker")
    process.ecalDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.ecalPreshowerDigis.sourceTag = cms.InputTag("rawDataRepacker")
    process.gctDigis.inputLabel = cms.InputTag("rawDataRepacker")
    process.gtDigis.DaqGtInputTag = cms.InputTag("rawDataRepacker")
    process.gtEvmDigis.EvmGtInputTag = cms.InputTag("rawDataRepacker")
    process.hcalDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.l1compare.FEDsourceEmul = cms.untracked.InputTag("rawDataRepacker")
    process.l1compare.FEDsourceData = cms.untracked.InputTag("rawDataRepacker")
    process.muonCSCDigis.InputObjects = cms.InputTag("rawDataRepacker")
    process.muonDTDigis.inputLabel = cms.InputTag("rawDataRepacker")
    process.muonRPCDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.scalersRawToDigi.scalersInputTag = cms.InputTag("rawDataRepacker")
    process.siPixelDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.siStripDigis.ProductLabel = cms.InputTag("rawDataRepacker")



### process customizations included here
from DQM.Integration.config.online_customizations_cfi import *
process = customise(process)
