import FWCore.ParameterSet.Config as cms

process = cms.Process("L1TEmuDQMlive")

process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.load("DQM.Integration.test.inputsource_cfi")
process.load("DQM.Integration.test.environment_cfi")

process.load("DQM.Integration.test.FrontierCondition_GT_cfi")
process.GlobalTag.RefreshEachRun = cms.untracked.bool(True)

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

process.load("DQM.L1TMonitor.L1TEmulatorMonitor_cff")    
process.load("DQM.L1TMonitorClient.L1TEMUMonitorClient_cff")    

#NL//this over-writting may be employed only when needed
#  ie quick module disabling, before new tags can be corrected)
from L1Trigger.HardwareValidation.L1HardwareValidation_cff import *
l1compare.COMPARE_COLLS = [1, 1, 1, 1,  1, 1, 1, 1, 1, 0, 1, 1]
newHWSequence = cms.Sequence(deEcal+deHcal+deRct+deGct+deDt+deDttf+deCsc+deCsctf+deRpc+deGmt+deGt*l1compare)
process.globalReplace("L1HardwareValidation", newHWSequence)

#N//validate online, then migrate to cff for offline
process.load("SimCalorimetry.EcalTrigPrimProducers.ecalTriggerPrimitiveDigis_readDBOffline_cff")
#N//needs to be removed from here
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

process.EventStreamHttpReader.SelectEvents = cms.untracked.PSet(
    SelectEvents = cms.vstring("*")
)
process.EventStreamHttpReader.consumerName = 'L1TEMU DQM Consumer'
process.dqmEnv.subSystemFolder = 'L1TEMU'
