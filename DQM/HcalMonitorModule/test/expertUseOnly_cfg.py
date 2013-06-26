import FWCore.ParameterSet.Config as cms
from DQM.HcalMonitorModule.HcalMonitorModule_cfi import * # there's probably a better way to do this, once I discover the difference between import and load
from DQM.HcalMonitorClient.HcalMonitorClient_cfi import * # ditto

process = cms.Process("HCALDQM")
#----------------------------
# Event Source
#-----------------------------
process.maxEvents = cms.untracked.PSet(
    # Specify number of events over which to run
    input = cms.untracked.int32(1000)
    )

#Use this when running over .dat file  
#process.source = cms.Source("NewEventStreamFileReader",
#                            fileNames = cms.untracked.vstring('/store/data/GlobalCruzet3MW33/A/000/056/416/GlobalCruzet3MW33.00056416.0001.A.storageManager.0.0000.dat')
#)


# Use this when running over .root file
process.source = cms.Source("PoolSource",
                            #fileNames = cms.untracked.vstring('/store/data/CRUZET3/Cosmics/RAW/v1/000/051/199/EA1F908F-AD4E-DD11-8235-000423D6A6F4.root')
                            fileNames= cms.untracked.vstring('/store/data/Commissioning08/Cosmics/RAW/v1/000/064/103/2A983512-E18F-DD11-BE84-001617E30CA4.root')
                            )


# process.source = cms.Source("EventStreamHttpReader",
#                             sourceURL = cms.string('http://cmsmon:50082/urn:xdaq-application:lid=29'),
#                             consumerPriority = cms.untracked.string('normal'),
#                             max_event_size = cms.int32(7000000),
#                             consumerName = cms.untracked.string('Playback Source'),
#                             max_queue_depth = cms.int32(5),
#                             maxEventRequestRate = cms.untracked.double(12.0),
#                             SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('*')
#                                                               ),
#                             headerRetryInterval = cms.untracked.int32(3)
#                             )

#To run on HCAL local runs:  (you need to have access to /bigspool or modify the path to the datafile)	
# (This should work in python, but has not yet been tested as of 7 August 2008)
###process.source = cms.Source("HcalTBSource",
###                            fileNames = cms.untracked.vstring('file:/afs/cern.ch/user/s/stjohn/scratch0/USC_044790.root'),
###                            streams   = cms.untracked.vstring(#HBHEa,b,c:
###                                                              'HCAL_DCC700','HCAL_DCC701','HCAL_DCC702','HCAL_DCC703','HCAL_DCC704','HCAL_DCC705',
###                                                              'HCAL_DCC706','HCAL_DCC707','HCAL_DCC708','HCAL_DCC709','HCAL_DCC710','HCAL_DCC711',
###                                                              'HCAL_DCC712','HCAL_DCC713','HCAL_DCC714','HCAL_DCC715','HCAL_DCC716','HCAL_DCC717',
###                                                              #HF:
###                                                              'HCAL_DCC718','HCAL_DCC719','HCAL_DCC720','HCAL_DCC721','HCAL_DCC722','HCAL_DCC723',
###                                                              #HO:
###                                                              #'HCAL_DCC724','HCAL_DCC725','HCAL_DCC726','HCAL_DCC727','HCAL_DCC728','HCAL_DCC729',
###                                                              #'HCAL_DCC730','HCAL_DCC731',
###                                                              #'HCAL_Trigger','HCAL_SlowData'
###                                                              )
###                            )

#----------------------------
# DQM Environment
#-----------------------------
process.load("DQMServices.Core.DQM_cfg")

#replace DQMStore.referenceFileName = "Hcal_reference.root"
process.load("DQMServices.Components.DQMEnvironment_cfi")

process.DQM.collectorHost = 'myhost'
process.DQM.collectorPort = 9092
process.dqmSaver.convention = 'Online'
#replace dqmSaver.dirName          = "."
process.dqmSaver.producer = 'DQM'
process.dqmEnv.subSystemFolder = 'Hcal'
# optionally change fileSaving  conditions
# replace dqmSaver.prescaleLS =   -1
# replace dqmSaver.prescaleTime = -1 # in minutes
# replace dqmSaver.prescaleEvt =  -1

# For Hcal local run files, replace dqmSaver.saveByRun = 2 
process.dqmSaver.saveByRun = 1


#-----------------------------
# Hcal Conditions: from Global Conditions Tag 
#-----------------------------
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
# process.GlobalTag.connect = "frontier://(proxyurl=http://localhost:3128)(serverurl=http://frontier1.cms:8000/FrontierOnProd)(serverurl=http://frontier2.cms:8000/FrontierOnProd)(retrieve-ziplevel=0)/CMS_COND_21X_GLOBALTAG"
process.GlobalTag.connect = 'frontier://Frontier/CMS_COND_21X_GLOBALTAG'
process.GlobalTag.globaltag = 'CRZT210_V1::All' # or any other appropriate
process.prefer("GlobalTag")

process.load("FWCore.MessageLogger.MessageLogger_cfi")


#-----------------------------
# Hcal DQM Source, including SimpleReconstrctor
#-----------------------------
process.load("DQM.HcalMonitorModule.HcalMonitorModule_cfi")

process.load("EventFilter.HcalRawToDigi.HcalRawToDigi_cfi")
process.load("RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_hbhe_cfi")
process.load("RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_ho_cfi")
process.load("RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_hf_cfi")
process.load("RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_zdc_cfi")

# hcalMonitor configurable values -----------------------
process.hcalMonitor.debug = False

# Turn on/off individual hcalMonitor modules ------------
# Expert-level info :  Turn off everything except ExpertMonitor, EEUSMonitor
process.hcalMonitor.DataFormatMonitor   = False
process.hcalMonitor.DigiMonitor         = False
process.hcalMonitor.RecHitMonitor       = False
process.hcalMonitor.TrigPrimMonitor     = False
process.hcalMonitor.PedestalMonitor     = False
process.hcalMonitor.DeadCellMonitor     = False
process.hcalMonitor.HotCellMonitor      = False
process.hcalMonitor.LEDMonitor          = False
process.hcalMonitor.BeamMonitor         = False
process.hcalMonitor.CaloTowerMonitor    = False
process.hcalMonitor.MTCCMonitor         = False
process.hcalMonitor.HcalAnalysis        = False
process.hcalMonitor.ExpertMonitor       = True
process.hcalMonitor.EEUSMonitor         = True

# This takes the default cfg values from the hcalMonitor base class and applies them to the subtasks.
setHcalTaskValues(process.hcalMonitor)

#-----------------------------
# Scheduling
#-----------------------------
process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound', 
        'TooManyProducts', 
        'TooFewProducts')
    )

# We don't bother to run the client in expert mode right now

process.p = cms.Path(process.hcalDigis*process.horeco*process.hfreco*process.hbhereco*process.zdcreco*process.hcalMonitor*process.dqmEnv*process.dqmSaver)







