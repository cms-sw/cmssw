import FWCore.ParameterSet.Config as cms

## VarParsing
import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing ('analysis')
options.register('globalTag', 'auto:run3_data', options.multiplicity.singleton, options.varType.string, 'name of global tag')
options.setDefault('maxEvents', 100000)
options.register('initialSeed',-2,options.multiplicity.singleton,options.varType.int,'initialSeed')
options.parseArguments()

## Process
process = cms.Process('TEST')

process.options.wantSummary = False
process.options.numberOfStreams = 10
process.options.numberOfThreads = 10 
process.maxEvents.input = options.maxEvents

## Source
process.source = cms.Source('PoolSource',
    fileNames = cms.untracked.vstring(options.inputFiles)
)
process.source = cms.Source("EmptySource")

## MessageLogger
process.load('FWCore.MessageLogger.MessageLogger_cfi')
process.MessageLogger.cerr.FwkReport = cms.untracked.PSet(
    reportEvery = cms.untracked.int32(5000),
    limit = cms.untracked.int32(10000000)
)

## GlobalTag
#process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
#from Configuration.AlCa.GlobalTag import GlobalTag
#process.GlobalTag = GlobalTag(process.GlobalTag, options.globalTag, '')

process.PrescaleService = cms.Service("PrescaleService",
    forceDefault = cms.bool(True),
    lvl1DefaultLabel = cms.string('Test'),
    lvl1Labels = cms.vstring(
        'Test',            
    ),
    prescaleTable = cms.VPSet( (
        cms.PSet(
            pathName = cms.string('HLT_PreTest_v1'),
            prescales = cms.vuint32(
                100,
            )
        ),
       ))
                                      )
                                    
      
process.hltPreTest = cms.EDFilter("HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag("gtStage2Digis"),
                                  offset = cms.uint32(0),
                                  
)

if options.initialSeed>=-1:
    print(f"setting seed to {options.initialSeed}")
    process.hltPreTest.initialSeed = cms.int32(options.initialSeed)
    

process.HLT_PreTest_v1 = cms.Path(process.hltPreTest)

process.outMod = cms.OutputModule("PoolOutputModule",
                                  SelectEvents = cms.untracked.PSet(
                                      SelectEvents = cms.vstring('HLT_PreTest_v1')
                                  ),
                                  compressionAlgorithm = cms.untracked.string('LZMA'),
                                  compressionLevel = cms.untracked.int32(1),
                                  dataset = cms.untracked.PSet(
                                      dataTier = cms.untracked.string('GEN'),
                                      filterName = cms.untracked.string('')
                                  ),
                                  eventAutoFlushCompressedSize = cms.untracked.int32(20971520),
                                  fileName = cms.untracked.string(options.outputFile),
                                  outputCommands = cms.untracked.vstring("keep *"),
                                  splitLevel = cms.untracked.int32(0)
                )

process.outPath = cms.EndPath(process.outMod)
