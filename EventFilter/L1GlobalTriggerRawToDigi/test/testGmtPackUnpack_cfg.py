import FWCore.ParameterSet.Config as cms

process = cms.Process("GMTTEST")
# L1 GT EventSetup
process.load("L1TriggerConfig.L1GtConfigProducers.L1GtConfig_cff")

# L1 GMT EventSetup
process.load("L1TriggerConfig.L1ScalesProducers.L1MuTriggerScalesConfig_cff")

process.load("L1TriggerConfig.L1ScalesProducers.L1MuTriggerPtScaleConfig_cff")

process.load("L1TriggerConfig.L1ScalesProducers.L1MuGMTScalesConfig_cff")

process.load("L1TriggerConfig.GMTConfigProducers.L1MuGMTParametersConfig_cff")

# GMT Emulator
process.load("L1Trigger.GlobalMuonTrigger.gmtDigis_cff")

process.load("L1Trigger.GlobalTrigger.gtDigis_cfi")

process.load("EventFilter.L1GlobalTriggerRawToDigi.l1GtPack_cfi")

process.load("EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('l1GtPack',  ## DEBUG mode 

        'l1GtUnpack'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG'), ## DEBUG mode 

        DEBUG = cms.untracked.PSet( ## DEBUG mode, all messages  

            limit = cms.untracked.int32(-1)
        )
    ),
    destinations = cms.untracked.vstring('cout')
)

process.source = cms.Source("L1MuGMTHWFileReader",
    fileNames = cms.untracked.vstring('file:/afs/cern.ch/user/i/imikulec/big/CMSSW_1_6_0_DAQ3/src/L1Trigger/GlobalMuonTrigger/test/gmt_testfile.h4mu.dat')
)

process.dump1 = cms.EDAnalyzer("L1MuGMTDump",
    GMTInputTag = cms.untracked.InputTag("gmtDigis")
)

process.dump2 = cms.EDAnalyzer("L1MuGMTDump",
    GMTInputTag = cms.untracked.InputTag("l1GtUnpack")
)

process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('keep *'),
    fileName = cms.untracked.string('gmtPackUnpack.root')
)

process.p = cms.Path(process.gmtDigis*process.gtDigis*process.l1GtPack*process.l1GtUnpack*process.dump1*process.dump2)
process.outpath = cms.EndPath(process.out)
process.gmtDigis.DTCandidates = 'source:DT'
process.gmtDigis.CSCCandidates = 'source:CSC'
process.gmtDigis.RPCbCandidates = 'source:RPCb'
process.gmtDigis.RPCfCandidates = 'source:RPCf'
process.gmtDigis.Debug = 9
process.gmtDigis.BX_min = -1
process.gmtDigis.BX_max = 1
process.gmtDigis.BX_min_readout = -1
process.gmtDigis.BX_max_readout = 1
process.gtDigis.inputMask = [1, 0]
process.l1GtPack.ActiveBoardsMask = 0x0100

