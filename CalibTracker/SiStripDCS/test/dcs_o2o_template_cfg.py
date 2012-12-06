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

#This is not currently used in the O2O code, an ASCII file with the map is used instead...
#Will eventually get rid of it completely...
process.SiStripConfigDb = cms.Service("SiStripConfigDb",
    ConfDb = cms.untracked.string('cms_trk_r/PASSWORD@cms_omds_tunnel'),
    #TNS_ADMIN = cms.untracked.string('/afs/cern.ch/user/g/gbenelli/O2O/connection_files'),                                   
    TNS_ADMIN = cms.untracked.string('/nfshome0/popcondev/conddb'),
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
    #EDIT here with your favorite connection_files directory!
    #authPath = cms.string('/opt/cmssw/shifter/o2o_dcs/connection_files'),
    #authPath = cms.string('/exports/slc4/CMSSW/Development/Users/gbenelli/connection_files'),
    #authPath = cms.string('/afs/cern.ch/user/g/gbenelli/O2O/connection_files'),
    authPath = cms.string('/nfshome0/popcondev/conddb'),

    #The Tmin and Tmax indicated here drive the ManualO2O.py script setting the overall interval
    #By default this is broken into 1 hour O2O jobs (1 cmsRun cfg per hour interval)
    # Format for date/time vector:  year, month, day, hour, minute, second, nanosecond      
    Tmin = cms.vint32(2010, 8, 11,  4,  0, 0, 000),
    Tmax = cms.vint32(YEAR, MONTH, DAY, HOUR,  0, 0, 000),
    
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
    #Remember to change this to a Pixel list if you are testing the O2O code with Pixels before
    #the proper migration is done...
    DetIdListFile = cms.string('CalibTracker/SiStripCommon/data/SiStripDetInfo.dat'),

    # Threshold to consider an HV channel on
    HighVoltageOnThreshold = cms.double(0.97),

    # Leave empty if you want to use the db
    PsuDetIdMapFile = cms.string("CalibTracker/SiStripDCS/data/StripPSUDetIDMap_FromJan132010.dat"),
    #This excluded detids file is not currently used (it was needed when there were unmapped detids.
    ExcludedDetIdListFile = cms.string('')
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
    #Connection string when using a local sqlite db file:
    #connect = cms.string('sqlite_file:dbfile.db'),
    #Connection string to access the Offline DB via Frontier:
    connect = cms.string('frontier://PromptProd/CMS_COND_31X_STRIP'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('SiStripDetVOffRcd'),
        #Tag when using the output of the ManualO2O.py (fake tag)
        #tag = cms.string('SiStripDetVOff_Fake_31X')
        tag = cms.string('SiStripDetVOff_v1_offline')
    )),
    #Logconnect string to access the local sqlite logfile:
    #logconnect = cms.untracked.string('sqlite_file:logfile.db')
    #logconnect = cms.untracked.string('oracle://cms_orcon_prod/CMS_COND_31X_POPCONLOG')
    logconnect = cms.untracked.string('frontier://PromptProd/CMS_COND_31X_POPCONLOG')
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
        #Length in seconds of minimum deltaT for 2 consecutive IOVs in the original data to be considered separately and not be merged by the IOV reduction
        DeltaTmin = cms.uint32(15),
        #Length in seconds of the maximum time an IOV sequence can be (i.e. one can be compressing sequences up to 120 seconds long, after that a new IOV would be made)
        MaxIOVlength = cms.uint32(120)
    )                                        
)



# -----------------------------------------------------------------------------
# Specify the processes to be run.  Here, we only run one.
# -----------------------------------------------------------------------------

process.p = cms.Path(process.siStripPopConDetVOff)
    
