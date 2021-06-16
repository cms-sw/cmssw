from __future__ import print_function
# L1 Trigger DQM sequence (L1T)
#
#   authors previous versions - see CVS
#
#   V.M. Ghete 2011-05-25 revised version of L1 Trigger DQM


import FWCore.ParameterSet.Config as cms

process = cms.Process("DQM")


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
 
process.load("DQM.Integration.config.environment_cfi")
process.dqmEnv.subSystemFolder = 'L1T'
process.dqmSaver.tag = 'L1T'
process.dqmSaver.runNumber = options.runNumber
process.dqmSaverPB.tag = 'L1T'
process.dqmSaverPB.runNumber = options.runNumber

#
# references needed

# Condition for P5 cluster
process.load("DQM.Integration.config.FrontierCondition_GT_cfi")
process.GlobalTag.RefreshEachRun = True
# Condition for lxplus: change and possibly customise the GT
#from Configuration.AlCa.GlobalTag import GlobalTag as gtCustomise
#process.GlobalTag = gtCustomise(process.GlobalTag, 'auto:run2_data', '')

#process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

#-------------------------------------
# sequences needed for L1 trigger DQM
#

# standard unpacking sequence 
process.load("Configuration.StandardSequences.RawToDigi_Data_cff")    

# L1 Trigger sequences 

# l1tMonitor and l1tMonitorEndPathSeq
process.load("DQM.L1TMonitor.L1TMonitor_cff")    

# L1 trigger synchronization module - it uses also HltHighLevel filter
process.load("DQM.L1TMonitor.L1TSync_cff")    


# l1tMonitorClient and l1tMonitorClientEndPathSeq
process.load("DQM.L1TMonitorClient.L1TMonitorClient_cff")    

#-------------------------------------
# paths & schedule for L1 Trigger DQM
#

# TODO define a L1 trigger L1TriggerRawToDigi in the standard sequence 
# to avoid all these remove
process.rawToDigiPath = cms.Path(process.RawToDigi)
#
process.RawToDigi.remove("siPixelDigis")
process.RawToDigi.remove("siStripDigis")
process.RawToDigi.remove("scalersRawToDigi")
process.RawToDigi.remove("castorDigis")
# for GCT, unpack all five samples
process.gctDigis.numberOfGctSamplesToUnpack = 5

#if ( process.runType.getRunType() == process.runType.pp_run_stage1 or process.runType.getRunType() == process.runType.cosmic_run_stage1):
process.gtDigis.DaqGtFedId = 809
#else:
#    process.gtDigis.DaqGtFedId = cms.untracked.int32(813)
# 
process.l1tMonitorPath = cms.Path(process.l1tMonitorOnline)

# separate L1TSync path due to the use of the HltHighLevel filter
process.l1tSyncPath = cms.Path(process.l1tSyncHltFilter+process.l1tSync)

#
process.l1tMonitorClientPath = cms.Path(process.l1tMonitorClient)

#
process.l1tMonitorEndPath = cms.EndPath(process.l1tMonitorEndPathSeq)

#
process.l1tMonitorClientEndPath = cms.EndPath(process.l1tMonitorClientEndPathSeq)

#
process.dqmEndPath = cms.EndPath(
                                 process.dqmEnv *
                                 process.dqmSaver *
                                 process.dqmSaverPB
                                 )

#
process.schedule = cms.Schedule(process.rawToDigiPath,
                                process.l1tMonitorPath,
                                process.l1tSyncPath,
                                #process.l1tMonitorClientPath,
                                #process.l1tMonitorEndPath,
                                #process.l1tMonitorClientEndPath,
                                process.dqmEndPath
                                )

#---------------------------------------------

# examples for quick fixes in case of troubles 
#    please do not modify the commented lines
#


#
# turn on verbosity in L1TEventInfoClient
#
# process.l1tEventInfoClient.verbose = cms.untracked.bool(True)


# remove module(s) or system sequence from l1tMonitorPath
#        quality test disabled also
#
process.l1tMonitorOnline.remove(process.bxTiming)
process.l1tMonitorOnline.remove(process.l1tBPTX)

#process.l1tMonitorOnline.remove(process.l1tLtc)

process.l1tMonitorOnline.remove(process.l1tDttf)

process.l1tMonitorOnline.remove(process.l1tCsctf) 

process.l1tMonitorOnline.remove(process.l1tRpctf)

process.l1tMonitorOnline.remove(process.l1tGmt)

#process.l1tMonitorOnline.remove(process.l1tGt) 
process.l1tGt.HistFolder = "L1T/L1TGTTestCrate"

#process.l1tMonitorOnline.remove(process.l1ExtraDqmSeq)

process.l1tMonitorOnline.remove(process.l1tRate)

process.l1tMonitorOnline.remove(process.l1tRctRun1)

#process.l1tMonitorOnline.remove(process.l1tGctSeq)


# remove module(s) or system sequence from l1tMonitorEndPath
#
process.l1tMonitorEndPathSeq.remove(process.l1s)
process.l1tMonitorEndPathSeq.remove(process.l1tscalers)

#
process.schedule.remove(process.l1tSyncPath)

#--------------------------------------------------
# Heavy Ion Specific Fed Raw Data Collection Label
#--------------------------------------------------

print("Running with run type = ", process.runType.getRunType())
process.castorDigis.InputLabel = "rawDataCollector"
process.csctfDigis.producer = "rawDataCollector"
process.dttfDigis.DTTF_FED_Source = "rawDataCollector"
process.ecalDigis.cpu.InputLabel = "rawDataCollector"
process.ecalPreshowerDigis.sourceTag = "rawDataCollector"
process.gctDigis.inputLabel = "rawDataCollector"
process.gtDigis.DaqGtInputTag = "rawDataCollector"
process.gtEvmDigis.EvmGtInputTag = "rawDataCollector"
process.hcalDigis.InputLabel = "rawDataCollector"
process.muonCSCDigis.InputObjects = "rawDataCollector"
process.muonDTDigis.inputLabel = "rawDataCollector"
process.muonRPCDigis.InputLabel = "rawDataCollector"
process.scalersRawToDigi.scalersInputTag = "rawDataCollector"
process.siPixelDigis.cpu.InputLabel = "rawDataCollector"
process.siStripDigis.ProductLabel = "rawDataCollector"
process.bxTiming.FedSource = "rawDataCollector"
process.l1s.fedRawData = "rawDataCollector"
    
if (process.runType.getRunType() == process.runType.hi_run):
    process.castorDigis.InputLabel = "rawDataRepacker"
    process.csctfDigis.producer = "rawDataRepacker"
    process.dttfDigis.DTTF_FED_Source = "rawDataRepacker"
    process.ecalDigis.cpu.InputLabel = "rawDataRepacker"
    process.ecalPreshowerDigis.sourceTag = "rawDataRepacker"
    process.gctDigis.inputLabel = "rawDataRepacker"
    process.gtDigis.DaqGtInputTag = "rawDataRepacker"
    process.gtEvmDigis.EvmGtInputTag = "rawDataRepacker"
    process.hcalDigis.InputLabel = "rawDataRepacker"
    process.muonCSCDigis.InputObjects = "rawDataRepacker"
    process.muonDTDigis.inputLabel = "rawDataRepacker"
    process.muonRPCDigis.InputLabel = "rawDataRepacker"
    process.scalersRawToDigi.scalersInputTag = "rawDataRepacker"
    process.siPixelDigis.cpu.InputLabel = "rawDataRepacker"
    process.siStripDigis.ProductLabel = "rawDataRepacker"
    process.bxTiming.FedSource = "rawDataRepacker"
    process.l1s.fedRawData = "rawDataRepacker"

### process customizations included here
from DQM.Integration.config.online_customizations_cfi import *
process = customise(process)
