import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process("TEST")

options = VarParsing.VarParsing('standard')
options.register('mode', 'trivial', VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string,
                 'type of emulation')
options.register('fedId', 0, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int,
                 'emulated FED id')
options.register('debug', False, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int,
                 'debugging mode')
options.register('dumpFRD', False, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int,
                 'also dump the FEDRawData content')
options.register('numCaptureBlocks', 1, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int,
                 'number of capture blocks to emulate')
options.register('numECONDs', -1, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int,
                 'number of ECON-Ds to emulate')
options.register('numChannelsPerERx', 37, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int,
                 'number of channels enabled per ERx')
options.register('numERxsPerECOND', 12, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int,
                 'number of ERxs enabled per ECON-D')
options.register('activeECONDs', [], VarParsing.VarParsing.multiplicity.list, VarParsing.VarParsing.varType.int,
                 'list of ECON-Ds enabled')
options.register('ECONDsInPassthrough', [], VarParsing.VarParsing.multiplicity.list, VarParsing.VarParsing.varType.int,
                 'list of ECON-Ds in passthrough mode')
options.register('ECONDsInCharacterisation', [], VarParsing.VarParsing.multiplicity.list, VarParsing.VarParsing.varType.int,
                 'list of ECON-Ds in characterisation mode')
options.register('ECONDToTStatus', 3, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int,
                 'default ToT status bits (aka TcTp bits) value to be emulated')
options.register('randomActiveCaptureBlocks', False, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int,
                 'randomly activate capture blocks on emulation')
options.register('randomActiveECOND', False, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int,
                 'randomly activate ECOn-Ds on emulation')
options.register('storeOutput', False, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int,
                 'also store the output into an EDM file')
options.register('storeRAWOutput', False, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int,
                 'also store the RAW output into a streamer file')
options.register('storeEmulatorInfo', False, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int,
                 'also store the emulator metadata')
options.register('inputFiles',
                 'file:/eos/cms/store/group/dpg_hgcal/tb_hgcal/2023/labtest/module822/pedestal_run0.root',
                 VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string,
                 'input TB file')
options.maxEvents = 1  # number of events to emulate
options.output = 'output.root'  # output EDM file
options.secondaryOutput = 'output.raw'  # output streamer file
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

# steer the emulator part
process.hgcalEmulatedSlinkRawData.emulatorType = options.mode
if process.hgcalEmulatedSlinkRawData.emulatorType == 'hgcmodule':
    process.hgcalEmulatedSlinkRawData.inputs = cms.untracked.vstring(options.inputFiles)
process.hgcalEmulatedSlinkRawData.storeEmulatorInfo = bool(options.storeEmulatorInfo)

# steer the number of capture blocks
if options.randomActiveCaptureBlocks:
    from random import randint
    process.hgcalEmulatedSlinkRawData.slinkParams.numCaptureBlocks = randint(1, 50)  # randomise the number of capture blocks emulated
else:
    process.hgcalEmulatedSlinkRawData.slinkParams.numCaptureBlocks = options.numCaptureBlocks
print('S-link: number of capture blocks: {}'.format(
    process.hgcalEmulatedSlinkRawData.slinkParams.numCaptureBlocks.value()))

# steer the number (and/or list) of ECON-Ds per capture block
if options.numECONDs > 0:
    for i in range(options.numECONDs - len(process.hgcalEmulatedSlinkRawData.slinkParams.ECONDs)):
        process.hgcalEmulatedSlinkRawData.slinkParams.ECONDs.append(process.hgcalEmulatedSlinkRawData.slinkParams.ECONDs[0].clone())
    process.hgcalEmulatedSlinkRawData.slinkParams.checkECONDsLimits = False  # allows to mess with unconventional, high number
                                                                             # of ECON-Ds per capture block

econd_id = 0
for econd in process.hgcalEmulatedSlinkRawData.slinkParams.ECONDs:
    # must use 'cms.' python configuration types
    if options.randomActiveECOND:  # randomly turn on/off any ECON-D in capture block
        from random import getrandbits
        econd.active = cms.bool(bool(getrandbits(1)))
    else:  # use a fixed, user-steered list of ECON-Ds in capture block
        econd.active = cms.bool((econd_id in options.activeECONDs))
    econd.passthroughMode = cms.bool((econd_id in options.ECONDsInPassthrough))
    econd.characterisationMode = cms.bool((econd_id in options.ECONDsInCharacterisation))
    econd.enabledERxs = cms.vuint32([i for i in range(options.numERxsPerECOND)])
    econd.numChannelsPerERx = cms.uint32(options.numChannelsPerERx)
    econd.defaultToTStatus = cms.uint32(options.ECONDToTStatus)
    print('ECON-D {}: active? {}, enabled eRxs: {}, number of channels/eRx: {}, passthrough? {}, characterisation? {}'.format(
        econd_id, bool(econd.active),
        [i for i in econd.enabledERxs], econd.numChannelsPerERx.value(),
        bool(econd.passthroughMode), bool(econd.characterisationMode)))
    econd_id += 1

# steer the unpacker
process.hgcalDigis.src = cms.InputTag('hgcalEmulatedSlinkRawData')
process.hgcalDigis.fedIds = cms.vuint32(options.fedId)
process.hgcalDigis.maxCaptureBlock = process.hgcalEmulatedSlinkRawData.slinkParams.numCaptureBlocks
process.hgcalDigis.numERxsInECOND = options.numERxsPerECOND
process.hgcalDigis.captureBlockECONDMax = max(  # allows to mess with unconventional, high number of ECON-Ds per capture block
    process.hgcalDigis.captureBlockECONDMax,
    len([ec for ec in process.hgcalEmulatedSlinkRawData.slinkParams.ECONDs if ec.active]))

process.p = cms.Path(process.hgcalEmulatedSlinkRawData * process.hgcalDigis)

if options.dumpFRD:
    process.dump = cms.EDAnalyzer("DumpFEDRawDataProduct",
        label = cms.untracked.InputTag('hgcalEmulatedSlinkRawData'),
        feds = cms.untracked.vint32(options.fedId),
        dumpPayload = cms.untracked.bool(True)
    )
    process.p *= process.dump

process.outpath = cms.EndPath()

if options.storeOutput:
    process.output = cms.OutputModule("PoolOutputModule",
        fileName = cms.untracked.string(options.output),
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
        fileName = cms.untracked.string(options.secondaryOutput)
    )
    process.outpath += process.outputRAW
