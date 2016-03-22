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
import os

process = cms.Process("DCSO2O")

# -----------------------------------------------------------------------------
# Load our message logger
# -----------------------------------------------------------------------------
# process.load("CalibTracker.SiStripDCS.MessLogger_cfi")
process.MessageLogger = cms.Service( "MessageLogger",
                                     debugModules = cms.untracked.vstring( "*" ),
                                     cout = cms.untracked.PSet( threshold = cms.untracked.string( "DEBUG" ) ),
                                     destinations = cms.untracked.vstring( "cout" )
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
# Define our ModuleHVBuilder process.  
#
# Parameters:
#   onlineDB            : the connection string for the database
#   authPath            : set to be $HOME
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
    onlineDB = cms.string('oracle://cms_omds_lb/CMS_TRK_R'),
    authPath = cms.string(os.environ["HOME"]),

    #The Tmin and Tmax indicated here drive the ManualO2O.py script setting the overall interval
    #By default this is broken into 1 hour O2O jobs (1 cmsRun cfg per hour interval)
    # Format for date/time vector:  year, month, day, hour, minute, second, nanosecond      
    Tmin = cms.vint32(_TMIN_),
    Tmax = cms.vint32(_TMAX_),
    
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
    #Remember to change this to a Pixel list if you are testing the O2O code with Pixels before
    #the proper migration is done...
    DetIdListFile = cms.string('CalibTracker/SiStripCommon/data/SiStripDetInfo.dat'),

    # Threshold to consider an HV channel on
    HighVoltageOnThreshold = cms.double(0.97),

    # Leave empty if you want to use the db
    PsuDetIdMapFile = cms.string("CalibTracker/SiStripDCS/data/StripPSUDetIDMap_FromFeb2016.dat"),

    #This excluded detids file is not currently used (it was needed when there were unmapped detids.
    ExcludedDetIdListFile = cms.string('')
)

# -----------------------------------------------------------------------------
# Service to write our data to the sqlite db.  This service is 
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

process.load("CondCore.CondDB.CondDB_cfi")
process.siStripPopConDetVOff = cms.EDAnalyzer( "SiStripO2ODetVOff",
                                     process.CondDB,
                                     # Get the last IOV from conditionDatabase.
#                                     conditionDatabase = cms.string("oracle://cms_orcon_prod/CMS_CONDITIONS"),
                                     # Leave empty for manual restart (will then get the last IOV from sqlite condDbFile). 
                                     conditionDatabase = cms.string(""),
                                     condDbFile = cms.string("sqlite:%s" % "_DBFILE_"),
                                     targetTag = cms.string("_TAG_"),
                                     # max length (in hours) before a new IOV is started for the same payload (use -1 to disable this)
                                     maxTimeBeforeNewIOV = cms.untracked.int32(168)
                                     )

# -----------------------------------------------------------------------------
# Specify the processes to be run.  Here, we only run one.
# -----------------------------------------------------------------------------
process.p = cms.Path(process.siStripPopConDetVOff)
