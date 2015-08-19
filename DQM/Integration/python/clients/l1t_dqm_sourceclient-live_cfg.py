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
#
# for testing in lxplus
#process.load("DQM.Integration.config.fileinputsource_cfi")

#----------------------------
# DQM Environment
 
process.load("DQM.Integration.config.environment_cfi")
process.dqmEnv.subSystemFolder = 'L1T'
process.dqmSaver.tag = 'L1T'

#
# references needed
process.DQMStore.referenceFileName = "/dqmdata/dqm/reference/l1t_reference.root"

# Condition for P5 cluster
process.load("DQM.Integration.config.FrontierCondition_GT_cfi")
es_prefer_GlobalTag = cms.ESPrefer('GlobalTag')
process.GlobalTag.RefreshEachRun = cms.untracked.bool(True)
# Condition for lxplus
#process.load("DQM.Integration.config.FrontierCondition_GT_Offline_cfi") 

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
process.gctDigis.numberOfGctSamplesToUnpack = cms.uint32(5)

if ( process.runType.getRunType() == process.runType.pp_run_stage1 or process.runType.getRunType() == process.runType.cosmic_run_stage1):
    process.gtDigis.DaqGtFedId = cms.untracked.int32(809)
else:
    process.gtDigis.DaqGtFedId = cms.untracked.int32(813)
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
                                 process.dqmSaver
                                 )

#
process.schedule = cms.Schedule(process.rawToDigiPath,
                                process.l1tMonitorPath,
                                process.l1tSyncPath,
                                process.l1tMonitorClientPath,
                                process.l1tMonitorEndPath,
                                process.l1tMonitorClientEndPath,
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
#process.l1tMonitorOnline.remove(process.bxTiming)
process.l1tMonitorOnline.remove(process.l1tBPTX)

#process.l1tMonitorOnline.remove(process.l1tLtc)

#process.l1tMonitorOnline.remove(process.l1Dttf)

#process.l1tMonitorOnline.remove(process.l1tCsctf) 

#process.l1tMonitorOnline.remove(process.l1tRpctf)

#process.l1tMonitorOnline.remove(process.l1tGmt)

#process.l1tMonitorOnline.remove(process.l1tGt) 

#process.l1tMonitorOnline.remove(process.l1ExtraDqmSeq)

process.l1tMonitorOnline.remove(process.l1tRate)

#process.l1tMonitorOnline.remove(process.l1tRctSeq)

#process.l1tMonitorOnline.remove(process.l1tGctSeq)


# remove module(s) or system sequence from l1tMonitorEndPath
#
#process.l1tMonitorEndPathSeq.remove(process.l1s)
#process.l1tMonitorEndPathSeq.remove(process.l1tscalers)

#
process.schedule.remove(process.l1tSyncPath)

#--------------------------------------------------
# Heavy Ion Specific Fed Raw Data Collection Label
#--------------------------------------------------

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
process.muonCSCDigis.InputObjects = cms.InputTag("rawDataCollector")
process.muonDTDigis.inputLabel = cms.InputTag("rawDataCollector")
process.muonRPCDigis.InputLabel = cms.InputTag("rawDataCollector")
process.scalersRawToDigi.scalersInputTag = cms.InputTag("rawDataCollector")
process.siPixelDigis.InputLabel = cms.InputTag("rawDataCollector")
process.siStripDigis.ProductLabel = cms.InputTag("rawDataCollector")
process.bxTiming.FedSource = cms.untracked.InputTag("rawDataCollector")
process.l1s.fedRawData = cms.InputTag("rawDataCollector")
    
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
    process.muonCSCDigis.InputObjects = cms.InputTag("rawDataRepacker")
    process.muonDTDigis.inputLabel = cms.InputTag("rawDataRepacker")
    process.muonRPCDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.scalersRawToDigi.scalersInputTag = cms.InputTag("rawDataRepacker")
    process.siPixelDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.siStripDigis.ProductLabel = cms.InputTag("rawDataRepacker")
    process.bxTiming.FedSource = cms.untracked.InputTag("rawDataRepacker")
    process.l1s.fedRawData = cms.InputTag("rawDataRepacker")

### process customizations included here
from DQM.Integration.config.online_customizations_cfi import *
process = customise(process)
