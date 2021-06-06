#
# This configuration is intended to be used to study errors spotted in the L1TEMU DQM output
#
# It can be run over streamer files or root files and outputs only events that fail the comparison.
#
# Alex Tapper 8/9/10
#

process = cms.Process('GctErrorFilter')

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.enable = cms.untracked.bool(True)
process.MessageLogger.cout.threshold = cms.untracked.string('WARNING')
process.MessageLogger.debugModules = cms.untracked.vstring('*')

# For streamer files
#process.source = cms.Source("NewEventStreamFileReader",
#                                                        fileNames = cms.untracked.vstring(
#    "/store/streamer/Data/A/000/142/414/Data.00142414.0044.A.storageManager.06.0000.dat"
#    )
#                            )

# For root files
process.source = cms.Source ( "PoolSource",
                              fileNames = cms.untracked.vstring(
    "/store/data/Commissioning10/MinimumBias/RAW/v4/000/135/244/B080D685-8A5C-DF11-9C93-001D09F251B8.root"
    )
                              )

# Number of events
process.maxEvents = cms.untracked.PSet ( input = cms.untracked.int32 ( 10000 ) )

# Global tag
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'GR10_P_V5::All'

# GCT emulator
import L1Trigger.GlobalCaloTrigger.gctDigis_cfi
process.valGctDigis = L1Trigger.GlobalCaloTrigger.gctDigis_cfi.gctDigis.clone()
process.valGctDigis.inputLabel = cms.InputTag("l1GctHwDigis")
process.valGctDigis.writeInternalData = cms.bool(True)
process.valGctDigis.preSamples = cms.uint32(0)
process.valGctDigis.postSamples = cms.uint32(0)
process.valGctDigis.useImprovedTauAlgorithm = cms.bool(True)

# GCT unpacker
process.load('EventFilter.GctRawToDigi.l1GctHwDigis_cfi')

# L1Comparator
process.load('L1Trigger.HardwareValidation.L1Comparator_cfi')
process.l1compare.GCTsourceData = cms.InputTag("l1GctHwDigis")
process.l1compare.COMPARE_COLLS = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
process.l1compare.DumpMode = cms.untracked.int32(1)
process.l1compare.DumpFile = cms.untracked.string('dump.txt')
process.l1compare.VerboseFlag = cms.untracked.int32(0)

# L1Comparator Filter    
process.load('L1Trigger.HardwareValidation.L1DEFilter_cfi')
process.l1defilter.DataEmulCompareSource = cms.InputTag("l1compare")
process.l1defilter.FlagSystems = cms.untracked.vuint32(0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0)

# Dump GCT digis
process.load('L1Trigger.L1GctAnalyzer.dumpGctDigis_cfi')
process.dumpGctDigis.doRctEm = cms.untracked.bool(False)
process.dumpGctDigis.doEm = cms.untracked.bool(False)
process.dumpGctDigis.doJets = cms.untracked.bool(True)
process.dumpGctDigis.doEmulated = cms.untracked.bool(True)
process.dumpGctDigis.emuGctInput = cms.untracked.InputTag("valGctDigis")
process.dumpGctDigis.doRegions = cms.untracked.bool(False)
process.dumpGctDigis.doInternEm = cms.untracked.bool(False)
process.dumpGctDigis.doEnergySums = cms.untracked.bool(True)
process.dumpGctDigis.doFibres = cms.untracked.bool(False)
process.dumpGctDigis.outFile = cms.untracked.string('gctDigis.txt')

# GCTErrorAnalyzer
process.load('L1Trigger.L1GctAnalyzer.gctErrorAnalyzer_cfi')

# Output ROOT file
process.TFileService = cms.Service("TFileService",
   fileName = cms.string( 'gctErrorAnalyzer.root' )
)

process.p = cms.Path(process.l1GctHwDigis*
                     process.valGctDigis*
                     process.l1compare*
                     ~process.l1defilter*
                     process.dumpGctDigis*
                     process.gctErrorAnalyzer)


process.output = cms.OutputModule( "PoolOutputModule",
                                   outputCommands = cms.untracked.vstring (
    "drop *",
    "keep *_*_*_GctErrorFilter",
    ),
                                   fileName = cms.untracked.string( "gctErrorFilter.root" ),
                                   SelectEvents = cms.untracked.PSet(
    SelectEvents = cms.vstring("p")
    )
                                   )

process.out = cms.EndPath( process.output )
