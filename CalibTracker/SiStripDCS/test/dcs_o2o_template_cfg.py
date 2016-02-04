# -----------------------------------------------------------------------------
# o2o-template_cfg.py : cmsRun configuration file for o2o DCS extraction
#
# Author  : Jo Cole
# Changes : Marco DeMattia
#           Dave Schudel
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Import configuration information & define our process
# -----------------------------------------------------------------------------
import FWCore.ParameterSet.Config as cms

process = cms.Process("test")

# -----------------------------------------------------------------------------
# Load our message logger
# -----------------------------------------------------------------------------
process.load("CalibTracker.SiStripDCS.MessLogger_cfi")
# -----------------------------------------------------------------------------
# Define our configuration database service.  
#
# Parameters:
#   ConfDB
#   TNS_ADMIN
#   UsingDb
#   Partitions
# -----------------------------------------------------------------------------

process.SiStripConfigDb = cms.Service("SiStripConfigDb",
    ConfDb = cms.untracked.string('cms_trk_r/PASSWORD@cms_omds_tunnel'),
    TNS_ADMIN = cms.untracked.string('/exports/slc4/CMSSW/Development/Users/gbenelli/connection_files'),
    UsingDb = cms.untracked.bool(True),
    Partitions = cms.untracked.PSet(
        PartTIBD = cms.untracked.PSet(
                PartitionName = cms.untracked.string("TI_13-JUN-2009_1"),
                ForceCurrentState = cms.untracked.bool(False),
                ForceVersions = cms.untracked.bool(True), 
                DcuPsuMapVersion = cms.untracked.vuint32(265,1)
                ),
        PartTOB = cms.untracked.PSet(
                PartitionName = cms.untracked.string("TO_30-JUN-2009_1"),
                ForceCurrentState = cms.untracked.bool(False),
                ForceVersions = cms.untracked.bool(True), 
                DcuPsuMapVersion = cms.untracked.vuint32(268,2)
                ),
        PartTECP = cms.untracked.PSet(
                PartitionName = cms.untracked.string("TP_09-JUN-2009_1"),
                ForceCurrentState = cms.untracked.bool(False),
                ForceVersions = cms.untracked.bool(True), 
                DcuPsuMapVersion = cms.untracked.vuint32(266,1)
                ),
        PartTECM = cms.untracked.PSet(
                PartitionName = cms.untracked.string("TM_09-JUN-2009_1"),
                ForceCurrentState = cms.untracked.bool(False),
                ForceVersions = cms.untracked.bool(True), 
                DcuPsuMapVersion = cms.untracked.vuint32(267,1)
                )
        )
)
# -----------------------------------------------------------------------------
# These lines are needed to run an EDAnalyzer without events.  We need to
#   specify an "EmptySource" so it doesn't try to load analysis data when it
#   starts up.  The maxEvents is set to 1 here - this tells the program how
#   many times to call the analyze() method in the EDAnalyzer.
# -----------------------------------------------------------------------------
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(1)
)


# -----------------------------------------------------------------------------
# Database Setup
# -----------------------------------------------------------------------------
process.load("CondCore.DBCommon.CondDBCommon_cfi")

process.CondDBCommon.connect = cms.string('oracle://cms_omds_tunnel/CMS_TRK_R')

# -----------------------------------------------------------------------------
# Define our ModuleHVBuilder process.  
#
# Parameters:
#   onlineDB            : the connection string for the database.  In  
#                           o2o-template_cfg.py, we save it as 'oracle://cms_omds_nolb/CMS_TRK_DCS_PVSS_COND' - it's
#                           converted to the correct value from the script
#                           run_o2o.sh (so we don't save connection info here)
#   authPath            : <unknown>
#   Tmin                : start date & time to extract data
#   Tmax                : end date & time to extract data
#   TSetMin             : <unknown>
#   queryType           : not needed any more - used to specify LASTVALUE or
#                           STATUSCHANGE query.  We use STATUSCHANGE now.
#   lastValueFile       : not needed
#   lastValueFromFile   : 
#   debugModeOn         : sets debug flag
# -----------------------------------------------------------------------------

process.SiStripDetVOffBuilder = cms.Service(
    "SiStripDetVOffBuilder",
    onlineDB = cms.string('oracle://cms_omds_tunnel/CMS_TRK_R'),
    #authPath = cms.string('/opt/cmssw/shifter/o2o_dcs/connection_files'),
    authPath = cms.string('/exports/slc4/CMSSW/Development/Users/gbenelli/connection_files'),

    # Format for date/time vector:  year, month, day, hour, minute, second, nanosecond      
    Tmin = cms.untracked.vint32(2009, 11, 23,  4,  0, 0, 000),
    Tmax = cms.untracked.vint32(2009, 11, 23,  16,  0, 0, 000),
    
    # Do NOT change this unless you know what you are doing!
    TSetMin = cms.vint32(2007, 11, 26, 0, 0, 0, 0),                                             
    
    # queryType can be either STATUSCHANGE or LASTVALUE                                
    queryType = cms.string('STATUSCHANGE'),
    
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
    PsuDetIdMapFile = cms.string("CalibTracker/SiStripDCS/data/PsuDetIdMap.dat"),
    ExcludedDetIdListFile = cms.string('CalibTracker/SiStripDCS/data/ExcludedSiStripDetInfo.dat')
)

# -----------------------------------------------------------------------------
# Service to write our data to the sqlite db (or Oracle).  This service is 
#   called from the endJob() method of the PopConAnalyzer class (which we have 
#   as SiStripPopConModuleHV) - that's why you won't find a call to it in the
#   DCS code.
#
# Parameters: (need to document..)
#   messageLevel
#   authenticationPath
#   timetype
#   connect
#   toPut
#     record
#     tag
#   logconnect
# -----------------------------------------------------------------------------

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(0),
        authenticationPath = cms.untracked.string('/nfshome0/popcondev/conddb')
    ),
    timetype = cms.untracked.string('timestamp'),
    connect = cms.string('sqlite_file:dbfile.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('SiStripDetVOffRcd'),
        tag = cms.string('SiStripDetVOff_Fake_31X')
    )),
    logconnect = cms.untracked.string('sqlite_file:logfile.db')
)

# -----------------------------------------------------------------------------
# Define a process: Here, we use a descendent of an EDAnalyzer
#   (a PopConAnalyzer)
#
# Parameters:
#   record
#   loggingOn
#   SinceAppendMode
#   Source
#     name
# -----------------------------------------------------------------------------

process.siStripPopConDetVOff = cms.EDAnalyzer("SiStripPopConDetVOff",
    record = cms.string('SiStripDetVOffRcd'),
    loggingOn= cms.untracked.bool(True),
    SinceAppendMode=cms.bool(True),
    Source = cms.PSet(
        DeltaTmin = cms.uint32(15),
        MaxIOVlength = cms.uint32(120)
    )                                        
)



# -----------------------------------------------------------------------------
# Specify the processes to be run.  Here, we only run one.
# -----------------------------------------------------------------------------

process.p = cms.Path(process.siStripPopConDetVOff)
    
