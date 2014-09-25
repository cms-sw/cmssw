import FWCore.ParameterSet.Config as cms

DBFILE = "MFConfig_090322_2pi_scaled"


process = cms.Process("DumpToDB")

process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(300000)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)


process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "START72_V1::All"
#process.GlobalTag.pfnPrefix = cms.untracked.string('frontier://FrontierProd/')

# VDrift, TTrig, TZero, Noise or channels Map into DB
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          process.CondDBSetup,
                                          connect = cms.string("sqlite_file:"+DBFILE+".db"),
                                          toPut = cms.VPSet(cms.PSet(record = cms.string("MagFieldConfigRcd"),
                                                                     tag = cms.string("MagFieldConfig_test"))))



from MagneticField.Engine.ScalingFactors_090322_2pi_090520_cfi import *

#Module to dump a file into a DB
process.dumpToDB = cms.EDAnalyzer("MagFieldConfigDBWriter",
    fieldScaling,
###    label = cms.untracked.string(''),
###    debugBuilder = cms.untracked.bool(False),
###    valueOverride = cms.int32(-1),
    # Parameters to be moved to the config DB
    version = cms.string('grid_1103l_090322_3_8t'),
    geometryVersion = cms.int32(90322),
    paramLabel = cms.string('parametrizedField'), #FIXME
    paramData = cms.vdouble(3.8),#FIXME
###    cacheLastVolume = cms.untracked.bool(True),#FIXME
                                                    
    gridFiles = cms.VPSet(
        cms.PSet( # Default tables, replicate sector 1
            volumes   = cms.string('1-312'),
            sectors   = cms.string('0') ,
            master    = cms.int32(1),
            path      = cms.string('grid.[v].bin'),
        ),

        cms.PSet( # Specific volumes in Barrel, sector 3
            volumes   = cms.string('176-186,231-241,286-296'),
            sectors   = cms.string('3') ,
            master    = cms.int32(3),
            path      = cms.string('S3/grid.[v].bin'),
        ),

        cms.PSet( # Specific volumes in Barrel, sector 4
            volumes   = cms.string('176-186,231-241,286-296'),
            sectors   = cms.string('4') ,
            master    = cms.int32(4),
            path      = cms.string('S4/grid.[v].bin'),
        ),

        cms.PSet(  # Specific volumes in Barrel and endcaps, sector 9
            volumes   = cms.string('14,15,20,21,24-27,32,33,40,41,48,49,56,57,62,63,70,71,286-296'),
            sectors   = cms.string('9') ,
            master    = cms.int32(9),
            path      = cms.string('S9/grid.[v].bin'),
        ),

        cms.PSet(  # Specific volumes in Barrel and endcaps, sector 10
            volumes   = cms.string('14,15,20,21,24-27,32,33,40,41,48,49,56,57,62,63,70,71,286-296'),
            sectors   = cms.string('10') ,
            master    = cms.int32(10),
            path      = cms.string('S10/grid.[v].bin'),
        ),
                                                        
        cms.PSet( # Specific volumes in Barrel and endcaps, sector 11
            volumes   = cms.string('14,15,20,21,24-27,32,33,40,41,48,49,56,57,62,63,70,71,286-296'),
            sectors   = cms.string('11') ,
            master    = cms.int32(11),
            path      = cms.string('S11/grid.[v].bin'),
        ),
    )
)
                                

process.p = cms.Path(process.dumpToDB)
    

