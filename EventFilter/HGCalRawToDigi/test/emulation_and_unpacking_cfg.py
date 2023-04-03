import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process("TEST")

options = VarParsing.VarParsing('analysis')
options.register('mode', 'empty', VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string, 'type of emulation')
options.register('fedId', 0, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int, 'emulated FED id')
options.register('debug', True, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int, 'debugging mode')
options.register('dumpFedRawData', False, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int, 'also dump the FEDRawData content')
options.register('numChannelsPerERx', 37, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int, 'number of channels enabled per ERx')
options.register('activeECONDs', [], VarParsing.VarParsing.multiplicity.list, VarParsing.VarParsing.varType.int, 'list of ECON-Ds enabled')
options.register('storeOutput', False, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int, 'also store the output into an EDM file')
options.register('storeRAWOutput', False, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int, 'also store the RAW output into a streamer file')
options.register('storeEmulatorInfo', False, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int, 'also store the emulator metadata')
options.inputFiles = 'file:/eos/cms/store/group/dpg_hgcal/tb_hgcal/2023/labtest/module822/pedestal_run0.root'
options.maxEvents = 20
options.secondaryOutputFile = 'output.raw'
options.parseArguments()

process.load('EventFilter.HGCalRawToDigi.hgcalEmulatedSlinkRawData_cfi')
process.load('EventFilter.HGCalRawToDigi.hgcalDigis_cfi')

process.load("FWCore.MessageService.MessageLogger_cfi")
if options.debug:
    process.MessageLogger.cerr.threshold = "DEBUG"
    process.MessageLogger.debugModules = ["hgcalEmulatedSlinkRawData", "hgcalDigis"]

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(options.maxEvents))
process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    hgcalEmulatedSlinkRawData = cms.PSet(initialSeed = cms.untracked.uint32(42))
)

process.source = cms.Source("EmptySource")
process.hgcalEmulatedSlinkRawData.emulatorType = options.mode
process.hgcalEmulatedSlinkRawData.econdParams.numChannelsPerERx = options.numChannelsPerERx
if process.hgcalEmulatedSlinkRawData.emulatorType == 'trivial':
    process.hgcalEmulatedSlinkRawData.slinkParams.activeECONDs = cms.vuint32(options.activeECONDs)
elif process.hgcalEmulatedSlinkRawData.emulatorType == 'hgcmodule':
    process.hgcalEmulatedSlinkRawData.inputs = cms.untracked.vstring(options.inputFiles)
process.hgcalEmulatedSlinkRawData.storeEmulatorInfo = bool(options.storeEmulatorInfo)
process.hgcalDigis.src = cms.InputTag('hgcalEmulatedSlinkRawData')
process.hgcalDigis.fedIds = cms.vuint32(options.fedId)

process.p = cms.Path(process.hgcalEmulatedSlinkRawData * process.hgcalDigis)

if options.dumpFedRawData:
    process.dump = cms.EDAnalyzer("DumpFEDRawDataProduct",
        label = cms.untracked.InputTag('hgcalEmulatedSlinkRawData'),
        feds = cms.untracked.vint32(options.fedId),
        dumpPayload = cms.untracked.bool(True)
    )
    process.p *= process.dump

process.outpath = cms.EndPath()

if options.storeOutput:
    process.output = cms.OutputModule("PoolOutputModule",
        fileName = cms.untracked.string(options.outputFile),
        outputCommands = cms.untracked.vstring(
            'drop *',
            'keep *_hgcalEmulatedSlinkRawData_*_*',
            'keep *_hgcalDigis_*_*',
        )
    )
    process.outpath += process.output

if options.storeRAWOutput:
    process.outputRAW = cms.OutputModule("FRDOutputModule",
        source = cms.InputTag('hgcalEmulatedSlinkRawData'),
        frdVersion = cms.untracked.uint32(6),
        frdFileVersion = cms.untracked.uint32(1),
        fileName = cms.untracked.string(options.secondaryOutputFile)
    )
    process.outpath += process.outputRAW
