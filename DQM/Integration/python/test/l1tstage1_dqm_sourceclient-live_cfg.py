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
process.load("DQM.Integration.test.inputsource_cfi")
#
# for testing in lxplus
#process.load("DQM.Integration.test.fileinputsource_cfi")

#----------------------------
# DQM Environment
#
 
#
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.dqmEnv.subSystemFolder = 'L1TStage1'

#
process.load("DQM.Integration.test.environment_cfi")
process.dqmSaver.dirName = '.'
#
# references needed
process.DQMStore.referenceFileName = "/dqmdata/dqm/reference/l1t_reference.root"

# Condition for P5 cluster
process.load("DQM.Integration.test.FrontierCondition_GT_cfi")
es_prefer_GlobalTag = cms.ESPrefer('GlobalTag')
process.GlobalTag.RefreshEachRun = cms.untracked.bool(True)
# Condition for lxplus
#process.load("DQM.Integration.test.FrontierCondition_GT_Offline_cfi") 

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

process.gtDigis.DaqGtFedId = cms.untracked.int32(809)
# 

# separate L1TSync path due to the use of the HltHighLevel filter
process.l1tSyncPath = cms.Path(process.l1tSyncHltFilter+process.l1tSync)

#

process.l1tMonitorPath = cms.Path(process.l1tMonitorStage1Online)
process.l1tMonitorClientPath = cms.Path(process.l1tMonitorStage1Client)
# Update HfRing thresholds to accomodate di-iso tau trigger thresholds
from L1TriggerConfig.L1ScalesProducers.l1CaloScales_cfi import l1CaloScales
l1CaloScales.L1HfRingThresholds = cms.vdouble(0.0, 24.0, 28.0, 32.0, 36.0, 40.0, 44.0, 48.0)
l1CaloScales.L1HtMissThresholds = cms.vdouble(
    0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
    0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19,
    0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29,
    0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39,
    0.40, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49,
    0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59,
    0.60, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69,
    0.70, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79,
    0.80, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89,
    0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99,
    1.00, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09,
    1.10, 1.11, 1.12, 1.13, 1.14, 1.15, 1.16, 1.17, 1.18, 1.19,
    1.20, 1.21, 1.22, 1.23, 1.24, 1.25, 1.26, 1.27
    )

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
process.l1tMonitorStage1Online.remove(process.bxTiming)
#process.l1tMonitorStage1Online.remove(process.l1tBPTX)

#process.l1tMonitorOnline.remove(process.l1tLtc)

process.l1tMonitorStage1Online.remove(process.l1tDttf)

process.l1tMonitorStage1Online.remove(process.l1tCsctf) 

process.l1tMonitorStage1Online.remove(process.l1tRpctf)

process.l1tMonitorStage1Online.remove(process.l1tGmt)

#process.l1tMonitorOnline.remove(process.l1tGt)
process.l1tGt.HistFolder = cms.untracked.string("L1TStage1/L1TStage1GT") 

#process.l1tMonitorOnline.remove(process.l1ExtraDqmSeq)
process.l1ExtraDQMStage1.DirName=cms.string("L1TStage1/L1ExtraStage1")

#process.l1tMonitorStage1Online.remove(process.l1tRate)

process.l1tMonitorStage1Online.remove(process.l1tRctSeq)

#process.l1tMonitorOnline.remove(process.l1tGctSeq)


# remove module(s) or system sequence from l1tMonitorEndPath
#
#process.l1tMonitorEndPathSeq.remove(process.l1s)

process.l1tMonitorEndPathSeq.remove(process.l1tscalers)
process.l1s.dqmFolder = cms.untracked.string("L1TStage1/L1Stage1Scalers_SM") 

process.l1tMonitorStage1Client.remove(process.l1TriggerQualityTests)

process.l1tStage1Layer2Client.monitorDir = cms.untracked.string('L1TStage1/stage1layer2')

process.l1tsClient.dqmFolder = cms.untracked.string("L1TStage1/L1Stage1Scalers_SM")

process.l1TriggerStage1Clients.remove(process.l1tEventInfoClient)
#process.l1TriggerStage1Clients.remove(process.l1tsClient)
process.l1TriggerStage1Clients.remove(process.l1tDttfClient)
process.l1TriggerStage1Clients.remove(process.l1tCsctfClient)
process.l1TriggerStage1Clients.remove(process.l1tRpctfClient)
process.l1TriggerStage1Clients.remove(process.l1tGmtClient)
process.l1TriggerStage1Clients.remove(process.l1tOccupancyClient)
process.l1TriggerStage1Clients.remove(process.l1tTestsSummary)
process.l1TriggerStage1Clients.remove(process.l1tEventInfoClient)
#
process.schedule.remove(process.l1tSyncPath)

# 
# un-comment next lines in case you use the file for private tests on the playback server
# see https://twiki.cern.ch/twiki/bin/view/CMS/DQMTest for instructions
#
#process.dqmSaver.dirName = '.'
#process.dqmSaver.saveByRun = 1
#process.dqmSaver.saveAtJobEnd = True

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
from DQM.Integration.test.online_customizations_cfi import *
process = customise(process)
