import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

# Export or inport Ecal laser correction to/from the condition database or
# an sqlite file.
# 
# Usage: cmsRun ecalLaserCondTools_cfg.py [mode=MODE] [connect=CONNECT_STRING] [authenticationPath=AUTHENTICATION_PATH] [inputFiles=INPUT_FILE_1] [inputFiles=INPUT_FILE2 ...]
#

# Default options:
options = VarParsing.VarParsing()

default_file_format = 'hdf'
defaults = {
    'verbosity': 5,
    'mode': '%s_file_to_db' % default_file_format,
    'connect':  'sqlite:///output.sqlite',
    'authenticationPath': '/nfshome0/popcondev/conddb',
    'inputFiles': 'input.%s' % default_file_format,
    'tag': 'test',
    'gtag': '',
    'start': 0,
    'stop': 10,
    'firstTime': -1,
    'timeBetweenEvents': 1,
    'lastTime': -1,
    'maxEvents': 1,
}

options.register('verbosity',
                 defaults['verbosity'],
                 VarParsing.VarParsing.multiplicity.singleton, 
                 VarParsing.VarParsing.varType.int,
                 "Vebosity level")

options.register('mode', 
                 defaults['mode'],
                 VarParsing.VarParsing.multiplicity.singleton, 
                 VarParsing.VarParsing.varType.string,
                 "Running mode: db_to_ascii_file, ascii_file_to_db, or hdf_file_to_db (default: %s)" % defaults["mode"]);

options.register('connect',
                 defaults['connect'],
                 VarParsing.VarParsing.multiplicity.singleton, 
                 VarParsing.VarParsing.varType.string,
                 "Database connection string (default: %s)" % defaults["connect"]);

options.register('authenticationPath',
                 defaults['authenticationPath'],
                 VarParsing.VarParsing.multiplicity.singleton, 
                 VarParsing.VarParsing.varType.string,
                 "Path to the file with the database authentication credentials")

options.register('gtag',
                 defaults['gtag'],
                 VarParsing.VarParsing.multiplicity.singleton, 
                 VarParsing.VarParsing.varType.string,
                 "Global condition tag used when reading the database. Use empty string to use the default tag.")


options.register('tag',
                 defaults['tag'],
                 VarParsing.VarParsing.multiplicity.singleton, 
                 VarParsing.VarParsing.varType.string,
                 "Condition tag to use when filling the database")

options.register('inputFiles',
                 defaults['inputFiles'],
                 VarParsing.VarParsing.multiplicity.list, 
                 VarParsing.VarParsing.varType.string,
                 "List of input files for file-to-database modes")

options.register('firstTime',
                 defaults['firstTime'],
                 VarParsing.VarParsing.multiplicity.singleton, 
                 VarParsing.VarParsing.varType.int,
                 "Time in nanoseconds of the first event generated to extract the IOVs (see EmptySource parameters).")

options.register('timeBetween',
                 defaults['timeBetweenEvents'],
                 VarParsing.VarParsing.multiplicity.singleton, 
                 VarParsing.VarParsing.varType.int,
                 "Time in nanoseconds between two events generated to extract the IOVs (see EmptySource parameters).")

options.register('maxEvents',
                 defaults['maxEvents'],
                 VarParsing.VarParsing.multiplicity.singleton, 
                 VarParsing.VarParsing.varType.int,
                 "Number of events to generated. Use 1 for file-to-database mode")


#Parse options from the command line
options.parseArguments()

process = cms.Process("EcalLaserDB")

process.source = cms.Source("EmptySource",
                            firstTime = cms.untracked.uint64(1),
                            timeBetweenEvents = cms.untracked.uint64(1))

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(options.maxEvents)

)
process.load("CondCore.CondDB.CondDB_cfi")
process.CondDB.connect = options.connect
process.CondDB.DBParameters.authenticationPath = options.authenticationPath

if options.mode in [ 'hdf_file_to_db', 'ascii_file_to_db']:
    process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                              process.CondDB,
                                              toPut = cms.VPSet(cms.PSet(
                                                record = cms.string('EcalLaserAPDPNRatiosRcd'),
                                                tag = cms.string(options.tag),
                                                timetype = cms.untracked.string('timestamp')
                                              )))

if options.mode in ['db_to_ascii_file']:
    process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
    if options.gtag:
        process.GlobalTag.globaltag = option.gtag

    process.ecalConditions = cms.ESSource("PoolDBESSource",
                                          process.CondDB,
                                          #siteLocalConfig = cms.untracked.bool(True),
                                          toGet = cms.VPSet(
                                              cms.PSet(
                                                  record = cms.string('EcalLaserAPDPNRatiosRcd'),
                                                  tag = cms.string(options.tag),
                                              )))
                                              

process.load("CalibCalorimetry.EcalTrivialCondModules.EcalLaserCondTools_cfi")

process.ecalLaserCondTools.mode = options.mode
process.ecalLaserCondTools.inputFiles = options.inputFiles
process.ecalLaserCondTools.verbosity = options.verbosity

process.path = cms.Path(process.ecalLaserCondTools)
