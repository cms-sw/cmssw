# The following comments couldn't be translated into the new config version:

# services

import FWCore.ParameterSet.Config as cms

process = cms.Process("ecalde")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

#
#  DQM SERVICES
#
process.load("DQMServices.Core.DQM_cfg")

process.load("DQMServices.Components.test.dqm_onlineEnv_cfi")

#generate dummy digis (e-digis)
process.load("L1Trigger.HardwareValidation.L1DummyProducer_cfi")

#introduce bias (d-digis)
process.load("L1Trigger.HardwareValidation.L1EmulBias_cfi")

#d|e comparison
process.load("L1Trigger.HardwareValidation.L1Comparator_cfi")

#produce general de-dqm sources
process.load("DQM.L1TMonitor.L1TDEMON_cfi")

#produce ecal specific de-dqm sources
process.load("DQM.L1TMonitor.L1TdeECAL_cfi")

# deEcal client
process.load("DQM.L1TMonitorClient.L1TdeECALClient_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(5000)
)
process.source = cms.Source("EmptySource")

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        l1bias = cms.untracked.uint32(7863453),
        l1dummy = cms.untracked.uint32(1762349)
    )
)

process.p = cms.Path(process.l1dummy*process.l1bias*process.l1compare*process.l1demon*process.l1demonecal*process.l1tdeEcalseqClient*process.dqmEnv*process.dqmSaver)
process.dqmSaver.fileName = 'L1T'
process.dqmSaver.dirName = '.'
process.dqmSaver.isPlayback = True
process.dqmEnv.subSystemFolder = 'L1T'
process.l1dummy.VerboseFlag = 0
process.l1bias.VerboseFlag = 0
process.l1bias.ETPsource = 'l1dummy'
process.l1compare.DumpMode = 1
process.l1compare.VerboseFlag = 0
process.l1compare.ETPsourceEmul = 'l1dummy'
process.l1compare.ETPsourceData = 'l1bias'
process.l1dummy.DO_SYSTEM = [1, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 
    0, 0]
process.l1bias.DO_SYSTEM = [1, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 
    0, 0]
process.l1compare.COMPARE_COLLS = [1, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 
    0, 0]
process.l1demon.VerboseFlag = 0

