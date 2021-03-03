import FWCore.ParameterSet.Config as cms

process = cms.Process("analyzer2")

process.load("EventFilter.CSCTFRawToDigi.csctfunpacker_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.enable = cms.untracked.bool(True)
process.MessageLogger.cout.threshold = cms.untracked.string('INFO')
process.MessageLogger.debugModules = cms.untracked.vstring('*')

#process.source = cms.Source("PoolSource",
#    fileNames = cms.untracked.vstring('file:./qwe.root'),
#    skipEvents = cms.untracked.uint32(10200)
#)

process.source = cms.Source("DaqSource",
    readerPluginName = cms.untracked.string('CSCFileReader'),
    readerPset = cms.untracked.PSet(
        firstEvent  = cms.untracked.int32(0),
        tfDDUnumber = cms.untracked.int32(0),
        FED760 = cms.untracked.vstring('RUI00'),
        RUI00  = cms.untracked.vstring(#'qwe.raw_760'
'/raid0/gartner/data/run133877/RAW/csc_00133877_EmuRUI00_Monitor_602.raw',
'/raid0/gartner/data/run133877/RAW/csc_00133877_EmuRUI00_Monitor_603.raw',
'/raid0/gartner/data/run133877/RAW/csc_00133877_EmuRUI00_Monitor_604.raw',
'/raid0/gartner/data/run133877/RAW/csc_00133877_EmuRUI00_Monitor_605.raw',
'/raid0/gartner/data/run133877/RAW/csc_00133877_EmuRUI00_Monitor_606.raw',
'/raid0/gartner/data/run133877/RAW/csc_00133877_EmuRUI00_Monitor_607.raw',
'/raid0/gartner/data/run133877/RAW/csc_00133877_EmuRUI00_Monitor_608.raw',
'/raid0/gartner/data/run133877/RAW/csc_00133877_EmuRUI00_Monitor_609.raw',
'/raid0/gartner/data/run133877/RAW/csc_00133877_EmuRUI00_Monitor_610.raw',
'/raid0/gartner/data/run133877/RAW/csc_00133877_EmuRUI00_Monitor_611.raw',
'/raid0/gartner/data/run133877/RAW/csc_00133877_EmuRUI00_Monitor_612.raw',
'/raid0/gartner/data/run133877/RAW/csc_00133877_EmuRUI00_Monitor_613.raw',
'/raid0/gartner/data/run133877/RAW/csc_00133877_EmuRUI00_Monitor_614.raw',
'/raid0/gartner/data/run133877/RAW/csc_00133877_EmuRUI00_Monitor_615.raw',
'/raid0/gartner/data/run133877/RAW/csc_00133877_EmuRUI00_Monitor_616.raw',
'/raid0/gartner/data/run133877/RAW/csc_00133877_EmuRUI00_Monitor_617.raw',
'/raid0/gartner/data/run133877/RAW/csc_00133877_EmuRUI00_Monitor_618.raw',
'/raid0/gartner/data/run133877/RAW/csc_00133877_EmuRUI00_Monitor_619.raw',
'/raid0/gartner/data/run133877/RAW/csc_00133877_EmuRUI00_Monitor_620.raw',
'/raid0/gartner/data/run133877/RAW/csc_00133877_EmuRUI00_Monitor_621.raw',
'/raid0/gartner/data/run133877/RAW/csc_00133877_EmuRUI00_Monitor_622.raw',
'/raid0/gartner/data/run133877/RAW/csc_00133877_EmuRUI00_Monitor_623.raw',
'/raid0/gartner/data/run133877/RAW/csc_00133877_EmuRUI00_Monitor_624.raw',
'/raid0/gartner/data/run133877/RAW/csc_00133877_EmuRUI00_Monitor_625.raw',
'/raid0/gartner/data/run133877/RAW/csc_00133877_EmuRUI00_Monitor_626.raw',
'/raid0/gartner/data/run133877/RAW/csc_00133877_EmuRUI00_Monitor_627.raw',
'/raid0/gartner/data/run133877/RAW/csc_00133877_EmuRUI00_Monitor_628.raw',
'/raid0/gartner/data/run133877/RAW/csc_00133877_EmuRUI00_Monitor_629.raw',
'/raid0/gartner/data/run133877/RAW/csc_00133877_EmuRUI00_Monitor_630.raw',
'/raid0/gartner/data/run133877/RAW/csc_00133877_EmuRUI00_Monitor_631.raw',
'/raid0/gartner/data/run133877/RAW/csc_00133877_EmuRUI00_Monitor_632.raw',
'/raid0/gartner/data/run133877/RAW/csc_00133877_EmuRUI00_Monitor_633.raw',
'/raid0/gartner/data/run133877/RAW/csc_00133877_EmuRUI00_Monitor_634.raw',
'/raid0/gartner/data/run133877/RAW/csc_00133877_EmuRUI00_Monitor_635.raw',
'/raid0/gartner/data/run133877/RAW/csc_00133877_EmuRUI00_Monitor_636.raw',
'/raid0/gartner/data/run133877/RAW/csc_00133877_EmuRUI00_Monitor_637.raw',
'/raid0/gartner/data/run133877/RAW/csc_00133877_EmuRUI00_Monitor_638.raw',
'/raid0/gartner/data/run133877/RAW/csc_00133877_EmuRUI00_Monitor_639.raw',
'/raid0/gartner/data/run133877/RAW/csc_00133877_EmuRUI00_Monitor_640.raw'
    )
  )
)

process.load("EventFilter.CSCTFRawToDigi.csctfpacker_cfi")
process.csctfpacker.lctProducer = cms.InputTag("csctfunpacker")
process.csctfpacker.mbProducer  = cms.InputTag("csctfunpacker:DT")
process.csctfpacker.trackProducer = cms.InputTag("csctfunpacker")

import EventFilter.CSCTFRawToDigi.csctfunpacker_cfi
process.csctfDigis = EventFilter.CSCTFRawToDigi.csctfunpacker_cfi.csctfunpacker.clone()
process.csctfDigis.producer = cms.InputTag("csctfpacker:CSCTFRawData")

process.csctfanalyzer = cms.EDAnalyzer("CSCTFAnalyzer",
         mbProducer     = cms.untracked.InputTag("csctfDigis:DT"),
         lctProducer    = cms.untracked.InputTag("csctfDigis:"),
         trackProducer  = cms.untracked.InputTag("csctfDigis:"),
         statusProducer = cms.untracked.InputTag("csctfDigis:")
#         lctProducer    = cms.untracked.InputTag("null:"),
#         trackProducer  = cms.untracked.InputTag("null:"),
#         statusProducer = cms.untracked.InputTag("null:")

)

process.FEVT = cms.OutputModule("PoolOutputModule",
        fileName = cms.untracked.string("qwe.root"),
        outputCommands = cms.untracked.vstring("drop *","keep *_csctfunpacker_*_*")
)

process.cscdumper = cms.EDAnalyzer("CSCFileDumper",
    output = cms.untracked.string("./qwe.raw"),
    events = cms.untracked.string("4111097")
)

process.p = cms.Path(process.csctfunpacker*process.csctfpacker*process.csctfDigis*process.csctfanalyzer)
#process.p = cms.Path(process.csctfunpacker*process.FEVT)
#process.p = cms.Path(process.csctfunpacker*process.csctfanalyzer)
#process.p = cms.Path(process.csctfpacker*process.cscdumper)

