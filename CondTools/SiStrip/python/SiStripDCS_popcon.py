import datetime
import os
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

options = VarParsing.VarParsing()
options.register('delay'
                , 1  # default value
                , VarParsing.VarParsing.multiplicity.singleton
                , VarParsing.VarParsing.varType.int
                , "Time delay (in hours) for the O2O. The O2O then queries the PVSS DB from last IOV until (current hour - delay), ignoring minutes and seconds."
                  )
options.register('sourceConnection'
                , 'oracle://cms_omds_adg/CMS_TRK_R'  # default value
                , VarParsing.VarParsing.multiplicity.singleton
                , VarParsing.VarParsing.varType.string
                , "Connection string to the PVSS DB."
                  )
options.register('destinationConnection'
                , 'sqlite_file:SiStripDetVOff.db'  # default value
                , VarParsing.VarParsing.multiplicity.singleton
                , VarParsing.VarParsing.varType.string
                , "Connection string to the DB where payloads will be possibly written."
                  )
options.register('conddbConnection'
                , 'oracle://cms_orcon_adg/CMS_CONDITIONS'  # default value
                , VarParsing.VarParsing.multiplicity.singleton
                , VarParsing.VarParsing.varType.string
                , "Connection string to the DB from which the last IOV is read."
                  )
options.register('tag'
                , 'SiStripDetVOff_test'
                , VarParsing.VarParsing.multiplicity.singleton
                , VarParsing.VarParsing.varType.string
                , "Tag written in destinationConnection and finally appended in targetConnection."
                  )
options.parseArguments()

# convert delay to tmax
dt = datetime.datetime.utcnow() - datetime.timedelta(hours=options.delay)
tmax = [dt.year, dt.month, dt.day, dt.hour, 0, 0, 0]

# authentication path to the key file
authPath = os.environ['COND_AUTH_PATH'] if 'COND_AUTH_PATH' in os.environ else os.environ["HOME"]

process = cms.Process("SiStripDCSO2O")

process.MessageLogger = cms.Service( "MessageLogger",
                                     debugModules = cms.untracked.vstring( "*" ),
                                     cout = cms.untracked.PSet( threshold = cms.untracked.string( "DEBUG" ) ),
                                     destinations = cms.untracked.vstring( "cout" )
                                     )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(1)
)

# -----------------------------------------------------------------------------
process.SiStripDetVOffBuilder = cms.Service(
    "SiStripDetVOffBuilder",
    onlineDB=cms.string(options.sourceConnection),
    authPath=cms.string(authPath),

    # Format for date/time vector:  year, month, day, hour, minute, second, nanosecond      
    Tmin = cms.vint32(2016, 1, 1, 0, 0, 0, 0),
    Tmax = cms.vint32(tmax),

    # Do NOT change this unless you know what you are doing!
    TSetMin = cms.vint32(2007, 11, 26, 0, 0, 0, 0),
    
    # queryType can be either STATUSCHANGE or LASTVALUE                                
    queryType = cms.string('STATUSCHANGE'),
    
    #Length in seconds of minimum deltaT for 2 consecutive IOVs in the original data to be considered separately and not be merged by the IOV reduction
    DeltaTmin = cms.uint32(2),

    #Length in seconds of the maximum time an IOV sequence can be (i.e. one can be compressing sequences up to 120 seconds long, after that a new IOV would be made)
    MaxIOVlength = cms.uint32(90),

    # if reading lastValue from file put insert file name here                              
    lastValueFile = cms.string(''),
    
    # flag to show if you are reading from file for lastValue or not                              
    lastValueFromFile = cms.bool(False),
    
    # flag to toggle debug output
    debugModeOn = cms.bool(False),

    # DetIdFile
    DetIdListFile = cms.string('CalibTracker/SiStripCommon/data/SiStripDetInfo.dat'),

    # Threshold to consider an HV channel on
    HighVoltageOnThreshold = cms.double(0.97),

    # Leave empty if you want to use the db
    PsuDetIdMapFile = cms.string("CalibTracker/SiStripDCS/data/StripPSUDetIDMap_FromFeb2016.dat"),

    #This excluded detids file is not currently used (it was needed when there were unmapped detids.
    ExcludedDetIdListFile = cms.string('')
)

# -----------------------------------------------------------------------------
process.load("CondCore.CondDB.CondDB_cfi")
process.siStripPopConDetVOff = cms.EDAnalyzer( "SiStripO2ODetVOff",
                                     process.CondDB,
                                     # Get the last IOV from conditionDatabase.
                                     # Leave empty for manual restart (will then get the last IOV from sqlite condDbFile). 
                                     conditionDatabase = cms.string(options.conddbConnection),
                                     condDbFile = cms.string(options.destinationConnection),
                                     targetTag = cms.string(options.tag),
                                     # max length (in hours) before a new IOV is started for the same payload (use -1 to disable this)
                                     maxTimeBeforeNewIOV=cms.untracked.int32(168)
                                     )

process.p = cms.Path(process.siStripPopConDetVOff)
