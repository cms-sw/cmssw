#description
#-generate random digis (emul) 
#-introduce simple bias (data)
#-compare initial and modified digis
#-generate dqm sources

import FWCore.ParameterSet.Config as cms

process = cms.Process("testdummybiasde")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load("L1Trigger.HardwareValidation.L1DummyProducer_cfi")
process.load("L1Trigger.HardwareValidation.L1EmulBias_cfi")
process.load("L1Trigger.HardwareValidation.L1Comparator_cfi")
process.load("DQM.L1TMonitor.L1TDEMON_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

process.source = cms.Source("EmptySource")

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        l1bias = cms.untracked.uint32(7863453),
        l1dummy = cms.untracked.uint32(1762349)
    )
)

process.DQMStore = cms.Service("DQMStore")

process.outputEvents = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('keep *'),
    fileName = cms.untracked.string('l1fakebias.root')
)

process.l1dummy.VerboseFlag = 0
process.l1bias.VerboseFlag = 0
process.l1bias.ETPsource = 'l1dummy'
process.l1bias.HTPsource = 'l1dummy'
process.l1bias.RCTsource = 'l1dummy'
process.l1bias.GCTsource = 'l1dummy'
process.l1bias.DTPsource = 'l1dummy'
process.l1bias.DTFsource = 'l1dummy'
process.l1bias.CTPsource = 'l1dummy'
process.l1bias.CTFsource = 'l1dummy'
process.l1bias.CTTsource = 'l1dummy'
process.l1bias.RPCsource = 'l1dummy'
process.l1bias.LTCsource = 'l1dummy'
process.l1bias.GMTsource = 'l1dummy'
process.l1bias.GLTsource = 'l1dummy'
process.l1compare.DumpMode = 1
process.l1compare.VerboseFlag = 0
process.l1compare.ETPsourceEmul = 'l1dummy'
process.l1compare.ETPsourceData = 'l1bias'
process.l1compare.HTPsourceEmul = 'l1dummy'
process.l1compare.HTPsourceData = 'l1bias'
process.l1compare.RCTsourceEmul = 'l1dummy'
process.l1compare.RCTsourceData = 'l1bias'
process.l1compare.GCTsourceEmul = 'l1dummy'
process.l1compare.GCTsourceData = 'l1bias'
process.l1compare.DTPsourceEmul = 'l1dummy'
process.l1compare.DTPsourceData = 'l1bias'
process.l1compare.DTFsourceEmul = 'l1dummy'
process.l1compare.DTFsourceData = 'l1bias'
process.l1compare.CTPsourceEmul = 'l1dummy'
process.l1compare.CTPsourceData = 'l1bias'
process.l1compare.CTFsourceEmul = 'l1dummy'
process.l1compare.CTFsourceData = 'l1bias'
process.l1compare.CTTsourceEmul = 'l1dummy'
process.l1compare.CTTsourceData = 'l1bias'
process.l1compare.RPCsourceEmul = 'l1dummy'
process.l1compare.RPCsourceData = 'l1bias'
process.l1compare.LTCsourceEmul = 'l1dummy'
process.l1compare.LTCsourceData = 'l1bias'
process.l1compare.GMTsourceEmul = 'l1dummy'
process.l1compare.GMTsourceData = 'l1bias'
process.l1compare.GLTsourceEmul = 'l1dummy'
process.l1compare.GLTsourceData = 'l1bias'
process.l1dummy.DO_SYSTEM = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0]
process.l1bias.DO_SYSTEM = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0]
process.l1compare.COMPARE_COLLS = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0]


process.p = cms.Path(process.l1dummy*process.l1bias*process.l1compare*process.l1demon)

