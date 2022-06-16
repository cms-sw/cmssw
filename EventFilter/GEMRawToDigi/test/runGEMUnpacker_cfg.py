# Auto generated configuration file
# using:
# Revision: 1.19
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v
# with command line options: SingleElectronPt10_cfi.py -s GEN,SIM,DIGI,L1 --pileup=NoPileUp --geometry DB --conditions=auto:startup -n 1 --no_exec
import FWCore.ParameterSet.Config as cms


# options
import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing('analysis')
options.register('skipEvents',
                 0,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Number of events to skip")
options.register('streamer',
                 False,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 "Read input from streamer file")
options.register('debug',
                 True,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 "Enable debug data")
options.register('dumpRaw',
                 False,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 "Print RAW data")
options.register('dumpDigis',
                 False,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 "Print digis")
options.register('histos',
                 True,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 "Produce standard histograms")
options.register('edm',
                 True,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 "Produce EDM file")
options.register('json',
                 '',
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "JSON file with list of good lumi sections")
options.register('evtDisp',
                 False,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 'Produce histos for individual events')
options.register('trigger',
                 False,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 'Trigger the data')
options.register('reconstruct',
                 False,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 'Reconstruct the data')
options.register('feds',
                 [1467,1468],
                 VarParsing.VarParsing.multiplicity.list,
                 VarParsing.VarParsing.varType.int,
                 "List of FEDs")
options.register('unpackerLabel',
                 'rawDataCollector',
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Label for the GEM unpacker RAW input collection")
options.register('useB904Data',
                 False,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool)

options.parseArguments()



from Configuration.Eras.Era_Run3_cff import Run3
process = cms.Process('RECO',Run3)

process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.RecoSim_cff')
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('EventFilter.GEMRawToDigi.muonGEMDigis_cfi')
process.load('L1Trigger.L1TGEM.simGEMDigis_cff')
process.load('EventFilter.L1TRawToDigi.validationEventFilter_cfi')
process.load("CommonTools.UtilAlgos.TFileService_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(options.maxEvents)
)

# Input source
if (options.streamer) :
    process.source = cms.Source(
        "NewEventStreamFileReader",
        fileNames = cms.untracked.vstring(options.inputFiles),
        skipEvents=cms.untracked.uint32(options.skipEvents)
    )
else :
    process.source = cms.Source(
        "PoolSource",
        fileNames = cms.untracked.vstring(options.inputFiles),
        skipEvents=cms.untracked.uint32(options.skipEvents),
        ## this line is needed to run the unpacker on output from AMC13SpyReadout.py
        labelRawDataLikeMC = cms.untracked.bool(False)
    )

if (options.json):
    import FWCore.PythonUtilities.LumiList as LumiList
    process.source.lumisToProcess = LumiList.LumiList(filename = options.json).getVLuminosityBlockRange()

process.options = cms.untracked.PSet(
    SkipEvent = cms.untracked.vstring('ProductNotFound')
)

# enable debug message logging for our modules
if (options.dumpRaw):
    process.MessageLogger.files.infos = cms.untracked.PSet(INFO = cms.untracked.PSet(limit = cms.untracked.int32(0)))

if (options.debug):
    process.MessageLogger.debugModules = cms.untracked.vstring('*')
    process.MessageLogger.cerr.threshold = cms.untracked.string('DEBUG')


# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '112X_dataRun3_Prompt_v5', '')
## for the time being the mapping does not work with the data label. Use MC instead
if options.useB904Data:
    process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2022_realistic', '')

# dump raw data
process.dumpRaw = cms.EDAnalyzer(
    "DumpFEDRawDataProduct",
    token = cms.untracked.InputTag(options.unpackerLabel),
    feds = cms.untracked.vint32(options.feds),
    dumpPayload = cms.untracked.bool(options.dumpRaw)
)

# optional EDM file
process.output = cms.OutputModule(
    "PoolOutputModule",
    outputCommands = cms.untracked.vstring("keep *"),
    fileName = cms.untracked.string('output_edm.root')
)

process.muonGEMDigis.InputLabel = options.unpackerLabel
process.simMuonGEMPadDigis.InputCollection = 'muonGEMDigis'

## schedule and path definition
process.p1 = cms.Path(process.dumpRaw)
process.p2 = cms.Path(process.muonGEMDigis)
process.p3 = cms.Path(process.simMuonGEMPadDigis * process.simMuonGEMPadDigiClusters)
process.p4 = cms.Path(process.gemRecHits * process.gemSegments)
process.out = cms.EndPath(process.output)
process.endjob_step = cms.EndPath(process.endOfProcess)

process.schedule = cms.Schedule()

# enable RAW printout
if options.dumpRaw:
    process.schedule.extend([process.p1])

# always add unpacker
process.schedule.extend([process.p2])

# triggers
if options.trigger:
    process.schedule.extend([process.p3])

# reconstruct rechits and segments
if options.reconstruct:
    process.schedule.extend([process.p4])

if options.edm:
    process.schedule.extend([process.out])

process.schedule.extend([process.endjob_step])
