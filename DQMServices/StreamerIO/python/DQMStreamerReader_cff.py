import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

options = VarParsing.VarParsing('analysis')

options.register('runNumber',
                 100, # default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Run number.")

options.register('datafnPosition',
                 3, # default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Data filename position in the positional arguments array 'data' in json file.")

options.register('runInputDir',
                 '/build1/micius/OnlineDQM_sample/', # default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Directory where the DQM files will appear.")

options.register('streamLabel',
                 '_streamA', # default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Stream label used in json discovery.")

options.register('scanOnce',
                 False, # default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 "Don't repeat file scans: use what was found during the initial scan. EOR file is ignored and the state is set to 'past end of run'.")

options.register('minEventsPerLumi',
                 1, # default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Minimum number of events to process per lumisection.")

options.register('delayMillis',
                 500, # default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Number of milliseconds to wait between file checks.")

options.register('nextLumiTimeoutMillis',
                 -1, # default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Number of milliseconds to wait before switching to the next lumi section if the current is missing, -1 to disable.")

options.register('skipFirstLumis',
                 False, # default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 "Skip (and ignore the minEventsPerLumi parameter) for the files which have been available at the begining of the processing. ")

options.register('deleteDatFiles',
                 False, # default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 "Delete data files after they have been closed, in order to save disk space.")

options.register('endOfRunKills',
                 False, # default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 "Kill the processing as soon as the end-of-run file appears, even if there are/will be unprocessed lumisections.")

options.register('endOfRunKills',
                 False, # default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 "Kill the processing as soon as the end-of-run file appears, even if there are/will be unprocessed lumisections.")



options.parseArguments()

# Input source
DQMStreamerReader = cms.Source("DQMStreamerReader",
    # DQMFileIterator
    runNumber = cms.untracked.uint32(options.runNumber),
    runInputDir = cms.untracked.string(options.runInputDir),
    streamLabel = cms.untracked.string(options.streamLabel),
    scanOnce = cms.untracked.bool(options.scanOnce),
    datafnPosition = cms.untracked.uint32(options.datafnPosition),
    delayMillis = cms.untracked.uint32(options.delayMillis),
    nextLumiTimeoutMillis = cms.untracked.int32(options.nextLumiTimeoutMillis),
    # DQMStreamerReader
    SelectEvents = cms.untracked.vstring("*"),
    minEventsPerLumi = cms.untracked.int32(options.minEventsPerLumi),
    skipFirstLumis = cms.untracked.bool(options.skipFirstLumis),
    deleteDatFiles = cms.untracked.bool(options.deleteDatFiles),
    endOfRunKills  = cms.untracked.bool(options.endOfRunKills),
)
