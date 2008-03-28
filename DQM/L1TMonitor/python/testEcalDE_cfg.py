# The following comments couldn't be translated into the new config version:

# services

import FWCore.ParameterSet.Config as cms

process = cms.Process("ecalde")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

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

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10000)
)
process.source = cms.Source("EmptySource")

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        l1bias = cms.untracked.uint32(7863453),
        l1dummy = cms.untracked.uint32(1762349)
    )
)

process.ModuleWebRegistry = cms.Service("ModuleWebRegistry")

process.DaqMonitorROOTBackEnd = cms.Service("DaqMonitorROOTBackEnd")

process.LockService = cms.Service("LockService",
    labels = cms.untracked.vstring('source')
)

process.MonitorDaemon = cms.Service("MonitorDaemon",
    AutoInstantiate = cms.untracked.bool(False),
    DestinationAddress = cms.untracked.string('cmsdaquser0.cern.ch'),
    SendPort = cms.untracked.int32(9090),
    NameAsSource = cms.untracked.string('GlobalDQM'),
    UpdateDelay = cms.untracked.int32(10000),
    reconnect_delay = cms.untracked.int32(5)
)

process.outputEvents = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('keep *'),
    fileName = cms.untracked.string('ecaltest.root')
)

process.p = cms.Path(process.l1dummy*process.l1bias*process.l1compare*process.l1demon*process.l1demonecal)
process.outpath = cms.EndPath(process.outputEvents)
process.l1dummy.VerboseFlag = 0
process.l1bias.VerboseFlag = 0
process.l1bias.ETPsource = 'l1dummy'
process.l1compare.DumpMode = 1
process.l1compare.VerboseFlag = 0
process.l1compare.ETPsourceEmul = 'l1dummy'
process.l1compare.ETPsourceData = 'l1bias'
process.l1dummy.DO_SYSTEM = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
process.l1bias.DO_SYSTEM = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
process.l1compare.COMPARE_COLLS = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
process.l1demon.VerboseFlag = 0

