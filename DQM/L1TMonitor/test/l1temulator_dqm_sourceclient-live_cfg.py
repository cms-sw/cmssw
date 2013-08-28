import FWCore.ParameterSet.Config as cms

process = cms.Process("L1TEmuDQMlive")

process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")

#replace this from the live version, see below
#process.load("DQM.Integration.test.inputsource_cfi")
#process.load("DQM.Integration.test.environment_cfi")


process.load("DQM.Integration.test.FrontierCondition_GT_cfi")

process.load("Configuration.StandardSequences.Geometry_cff")

process.load("DQM.L1TMonitor.L1TEmulatorMonitor_cff")    
process.load("DQM.L1TMonitorClient.L1TEMUMonitorClient_cff")    

#NL//this over-writting may be employed only when needed
#  ie quick module disabling, before new tags can be corrected)
from L1Trigger.HardwareValidation.L1HardwareValidation_cff import *
##NL//ctp temporarily disabled (infinite time sorting too large collections)
#l1compare.COMPARE_COLLS = [1, 1, 1, 1,  1, 1, 0, 1, 1, 0, 1, 1]
#newHWSequence = cms.Sequence(deEcal+deHcal+deRct+deGct+deDt+deDttf+deCsc+deCsctf+deRpc+deGmt+deGt*l1compare)
#process.globalReplace("L1HardwareValidation", newHWSequence)

#N//needs to be removed from here
process.load("SimCalorimetry.EcalTrigPrimProducers.ecalTriggerPrimitiveDigis_readDBOffline_cff")
#process.load("SimCalorimetry.EcalTrigPrimProducers.ecalTriggerPrimitiveDigis_craft_cff")
#process.EcalTrigPrimESProducer.DatabaseFile = 'TPG_craft.txt.gz' 



## Subsystem masking in summary map (case insensitive):
## l1t: all, gt, muons, jets, taujets, isoem, nonisoem, met
process.l1temuEventInfoClient.dataMaskedSystems =cms.untracked.vstring("All")
## Available emulator masks (case insensitive):
## l1temul: "all"; "dttf", "dttpg", "csctf", "csctpg", "rpc", "gmt", "ecal", "hcal", "rct", "gct", "glt"
process.l1temuEventInfoClient.emulatorMaskedSystems = cms.untracked.vstring("dttf", "dttpg", "csctf", "csctpg", "rpc", "ecal", "hcal", "glt")

##no references needed
#replace DQMStore.referenceFileName = "L1TEMU_reference.root"

#process.EventStreamHttpReader.SelectEvents = cms.untracked.PSet(
#    SelectEvents = cms.vstring("*")
#)
#process.EventStreamHttpReader.consumerName = 'L1TEMU DQM Consumer'
process.dqmEnv.subSystemFolder = 'L1TEMU'


#### additions

process.DQMStore.verbose = 0
process.dqmSaver.convention = 'Online'
process.dqmSaver.dirName = '.'
process.dqmSaver.producer = 'DQM'
process.dqmEnv.subSystemFolder = 'L1TEMU'
process.dqmSaver.saveByRun = -1
process.dqmSaver.saveAtJobEnd = False

process.l1compare.DumpMode = -1
process.l1compare.VerboseFlag = 1
process.l1demon.VerboseFlag = 1
#process.l1demonecal.VerboseFlag = 1
#process.l1demongct.VerboseFlag = 1
#process.l1tderct.verbose = False
process.l1demon.disableROOToutput = False
process.l1demonecal.disableROOToutput = False
process.l1demongct.disableROOToutput = False
process.l1tderct.disableROOToutput = False
#process.l1demon.RunInFilterFarm=True

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)
#process.source = cms.Source("PoolSource",
process.source = cms.Source("NewEventStreamFileReader",
    fileNames = cms.untracked.vstring(
    #'/store/data/BeamCommissioning09/Cosmics/RAW/v1/000/120/020/CE4D0EF0-47CC-DE11-BF95-0030487C6062.root'
 'file:/lookarea_SM/Data.00129937.0001.A.storageManager.00.0000.dat'
    #'file:/lookarea_SM/MWGR_29.00106019.0036.A.storageManager.07.0000.dat'
    )
)

#ETP,HTP,RCT,GCT, DTP,DTF,CTP,CTF,RPC, LTC,GMT,GLT 
process.l1compare.COMPARE_COLLS = [0,0,0,0, 0,0,0,0,0, 0,1,0]
#process.l1compare.COMPARE_COLLS = [1,1,1,1, 1,1,1,0,1, 0,1,1]
