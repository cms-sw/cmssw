from __future__ import print_function
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
import os

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
                 "Database to read: 'TZeroDB', 'TTrigDB',  'VDriftDB',  or 'UncertDB'")

options.register('inputfile',
                 '', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Input text file to be converted")


options.parseArguments()

DBFORMAT  = options.dbformat
TYPE      = options.type
INPUTFILE = options.inputfile

#Input sanification

if DBFORMAT not in ['Legacy', 'DTRecoConditions'] :
    print('\nERROR: invalid value for dbformat: ',  DBFORMAT,'\n')
    exit()
    
if TYPE not in ['TZeroDB', 'TTrigDB',  'VDriftDB', 'UncertDB'] :
    print('\nERROR: invalid value for type: ',  TYPE,'\n')
    exit()

if INPUTFILE == '' :
    print('\nERROR: must specify inputfile\n')
    exit()
    


process = cms.Process("DumpFileToDB")
process.load("CondCore.DBCommon.CondDBSetup_cfi")


process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)



OUTPUTFILE = INPUTFILE.replace('.txt','')+"_"+DBFORMAT+".db"


RECORD=""
GRANULARITY = "bySL"

if TYPE=="TZeroDB" :
    RECORD = "DTT0Rcd"
    GRANULARITY = "byWire"
if DBFORMAT=="Legacy" :
    if TYPE=="TTrigDB" : RECORD = "DTTtrigRcd"
    if TYPE=="VDriftDB" : RECORD = "DTMtimeRcd"
    if TYPE=="UncertDB" :
        RECORD = "DTRecoUncertaintiesRcd"
        print('\nWARNING, Legacy RecoUncertDB is deprecated, as it is no longer used in reconstruction code')
elif DBFORMAT=="DTRecoConditions" :
    if TYPE=="TTrigDB" : RECORD = "DTRecoConditionsTtrigRcd"
    if TYPE=="VDriftDB" : RECORD = "DTRecoConditionsVdriftRcd"
    if TYPE=="UncertDB" :
        RECORD = "DTRecoConditionsUncertRcd"
        TYPE='RecoUncertDB'
try:
    os.remove(OUTPUTFILE)
except OSError:
    pass


print('\n Reading ', TYPE, ' from ', INPUTFILE)
print('      Record : ', RECORD)
print('writing db file : ', OUTPUTFILE, '\n')


process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          process.CondDBSetup,
                                          connect = cms.string("sqlite_file:"+OUTPUTFILE),
                                          toPut = cms.VPSet(cms.PSet(record = cms.string(RECORD),
                                                                     tag = cms.string(TYPE)))
                                          )



#Module to convert calibration table into a DB file
process.dumpToDB = cms.EDAnalyzer("DumpFileToDB",
                                  calibFileConfig = cms.untracked.PSet(
                                      calibConstFileName = cms.untracked.string(INPUTFILE),
                                      calibConstGranularity = cms.untracked.string(GRANULARITY),
                                      ),
                                  dbFormat = cms.untracked.string(DBFORMAT),
                                  dbToDump = cms.untracked.string(TYPE),
                                )




process.p = cms.Path(process.dumpToDB)
    

