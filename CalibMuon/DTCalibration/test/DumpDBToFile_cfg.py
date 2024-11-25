import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

options = VarParsing.VarParsing()

options.register('dbformat',
                 'Legacy', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "DB format to use: 'Legacy' or 'DTRecoConditions'")

options.register('type',
                 'TTrigDB', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Database to read: 'TZeroDB', 'TTrigDB',  'VDriftDB', 'UncertDB', 'NoiseDB', 'DeadDB', 'ChannelsDB'")

options.register('GT',
                 'auto:run2_data', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Global tag to read, default is auto:run2_data")

options.register('inputfile',
                 '', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Read payload from db file instead than GT")

options.register('inputtag',
                 '', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Read payload from tag in frontier instead than GT")

options.register('label',
                 '', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Label of the payload to read (e.g.: 'cosmics', 'ph2',  ...)")

options.register('run',
                 999999, #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Run number (determines IOV to be read)")

options.parseArguments()

DBFORMAT  = options.dbformat
TYPE      = options.type
INPUTFILE = options.inputfile
INPUTTAG  = options.inputtag
GLOBALTAG = options.GT
LABEL     = options.label
RUN       = options.run



#Input sanification

if DBFORMAT not in ['Legacy', 'DTRecoConditions'] :
    print('\nERROR: invalid value for dbformat: ',  DBFORMAT,'\n')
    exit()
    
if TYPE not in ['TZeroDB', 'TTrigDB',  'VDriftDB', 'UncertDB', 'NoiseDB', 'DeadDB', 'ChannelsDB'] :
    print('\nERROR: invalid value for type: ',  TYPE,'\n')
    exit()


if INPUTTAG!="" and INPUTFILE!="" :
    print('\nERROR: specify either inputtag or inputfile\n')
    exit()




ofExt = {'TZeroDB'  : '_t0.txt',
         'TTrigDB'  : '_ttrig.txt',
         'VDriftDB' : '_vdrift.txt',
         'UncertDB' : '_uncert.txt',
         'NoiseDB'  : '_noise.txt',
         'DeadDB'   : '_dead.txt',
         'ChannelsDB' : '_channels.txt'}


if INPUTFILE!="":
    OUTPUTFILE=INPUTFILE
elif INPUTTAG!="" :
    OUTPUTFILE=INPUTTAG
else :
    OUTPUTFILE=GLOBALTAG+"_"+str(RUN)
    if OUTPUTFILE[0:5] == 'auto:' : OUTPUTFILE=OUTPUTFILE[5:]
OUTPUTFILE+=ofExt[TYPE]


###########

RECORD=""
if TYPE=="TZeroDB" : RECORD = "DTT0Rcd"
elif DBFORMAT=="Legacy" :
    if TYPE=="TTrigDB" : RECORD = "DTTtrigRcd"
    if TYPE=="VDriftDB" : RECORD = "DTMtimeRcd"
    if TYPE=="UncertDB" :
        RECORD = ""
        print('\nERROR, Legacy RecoUncertDB is no longer supported')
elif DBFORMAT=="DTRecoConditions" :
    if TYPE=="TTrigDB" : RECORD = "DTRecoConditionsTtrigRcd"
    if TYPE=="VDriftDB" : RECORD = "DTRecoConditionsVdriftRcd"
    if TYPE=="UncertDB" : RECORD = "DTRecoConditionsUncertRcd"


if TYPE == 'NoiseDB' : RECORD = 'DTStatusFlagRcd'
elif TYPE == 'DeadDB' : RECORD = 'DTDeadFlagRcd'
elif TYPE == 'ChannelsDB' : RECORD = 'DTReadOutMappingRcd'

process = cms.Process("DumpDBToFile")
process.load("CondCore.CondDB.CondDB_cfi")

process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(RUN)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)


process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, GLOBALTAG, '')

# Read from local db file
if INPUTFILE!="" :
    print("\nDumpDBToFile: Read from: ", INPUTFILE)
    print("              Record:    ", RECORD)
    print("              Type:      ", TYPE)


    process.GlobalTag.toGet = cms.VPSet(
        cms.PSet(record = cms.string(RECORD),
                 tag = cms.string(TYPE), # NOTE: commonly used tags for db files are 'ttrig', 't0','T0DB', etc.
                 connect = cms.string("sqlite_file:"+INPUTFILE)
                 )
        )


# Read payload with the specified tag from frontier
if INPUTTAG!="" :
    print("\nDumpDBToFile: Read from Frontier, tag:    ", INPUTTAG)
    print("                               Record: ", RECORD)
    print("                                 Type:   ", TYPE)

    process.GlobalTag.toGet = cms.VPSet(
        cms.PSet(record = cms.string(RECORD),
                 tag = cms.string(INPUTTAG),
                 connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS'),
#                 connect = cms.string('frontier://FrontierProd/CMS_COND_DT_000')
                 )
        )

# Read payload specified in the GT
else :
    print("\nDumpDBToFile: Read from GT:", GLOBALTAG)
    print("                      Type:", TYPE)

print('Writing to file: ', OUTPUTFILE, '\n')




process.dumpT0ToFile = cms.EDAnalyzer("DumpDBToFile",
    dbToDump = cms.untracked.string('TZeroDB'),
    dbLabel = cms.untracked.string(LABEL),
    calibFileConfig = cms.untracked.PSet(
        nFields = cms.untracked.int32(8),
        calibConstGranularity = cms.untracked.string('byWire')
    ),
    outputFileName = cms.untracked.string(OUTPUTFILE)
)

process.dumpTTrigToFile = cms.EDAnalyzer("DumpDBToFile",
    dbToDump = cms.untracked.string('TTrigDB'),
    dbLabel = cms.untracked.string(LABEL),
    dbFormat = cms.untracked.string(DBFORMAT),
    calibFileConfig = cms.untracked.PSet(
        nFields = cms.untracked.int32(8),
        calibConstGranularity = cms.untracked.string('bySL')
    ),
    outputFileName = cms.untracked.string(OUTPUTFILE)
)


process.dumpVdToFile = cms.EDAnalyzer("DumpDBToFile",
    dbToDump = cms.untracked.string('VDriftDB'),
    dbLabel = cms.untracked.string(LABEL),
    dbFormat = cms.untracked.string(DBFORMAT),
    calibFileConfig = cms.untracked.PSet(
        nFields = cms.untracked.int32(8),
        calibConstGranularity = cms.untracked.string('bySL')
    ),
    outputFileName = cms.untracked.string(OUTPUTFILE)
)


process.dumpUncertToFile = cms.EDAnalyzer("DumpDBToFile",
    dbToDump = cms.untracked.string('RecoUncertDB'),
    dbLabel = cms.untracked.string(LABEL),
    dbFormat = cms.untracked.string(DBFORMAT),
    calibFileConfig = cms.untracked.PSet(
        nFields = cms.untracked.int32(8),
        calibConstGranularity = cms.untracked.string('bySL')
    ),
    outputFileName = cms.untracked.string(OUTPUTFILE)
)

process.dumpNoiseToFile = cms.EDAnalyzer("DumpDBToFile",
    dbToDump = cms.untracked.string('NoiseDB'),
    dbLabel = cms.untracked.string(LABEL),
    dbFormat = cms.untracked.string(DBFORMAT),
    calibFileConfig = cms.untracked.PSet(
        nFields = cms.untracked.int32(8),
        calibConstGranularity = cms.untracked.string('byWire')
    ),
    outputFileName = cms.untracked.string(OUTPUTFILE)
)

process.dumpDeadToFile = cms.EDAnalyzer("DumpDBToFile",
    dbToDump = cms.untracked.string('DeadDB'),
    dbLabel = cms.untracked.string(LABEL),
    dbFormat = cms.untracked.string(DBFORMAT),
    calibFileConfig = cms.untracked.PSet(
        nFields = cms.untracked.int32(8),
        calibConstGranularity = cms.untracked.string('byWire')
    ),
    outputFileName = cms.untracked.string(OUTPUTFILE)
)

process.dumpChannelsToFile = cms.EDAnalyzer("DumpDBToFile",
    dbToDump = cms.untracked.string('ChannelsDB'),
    dbLabel = cms.untracked.string(LABEL),
    dbFormat = cms.untracked.string(DBFORMAT),
    calibFileConfig = cms.untracked.PSet(
        nFields = cms.untracked.int32(8),
        calibConstGranularity = cms.untracked.string('byWire')
    ),
    outputFileName = cms.untracked.string(OUTPUTFILE)
)



if   TYPE=="TZeroDB" :     process.p2 = cms.Path(process.dumpT0ToFile)
elif TYPE=="TTrigDB" :     process.p2 = cms.Path(process.dumpTTrigToFile)
elif TYPE=="VDriftDB" :    process.p2 = cms.Path(process.dumpVdToFile)
elif TYPE=="UncertDB":     process.p2 = cms.Path(process.dumpUncertToFile)
elif TYPE=='NoiseDB' :     process.p2 = cms.Path(process.dumpNoiseToFile)
elif TYPE=='DeadDB' :      process.p2 = cms.Path(process.dumpDeadToFile)
elif TYPE=='ChannelsDB' :  process.p2 = cms.Path(process.dumpChannelsToFile)


