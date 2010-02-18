import FWCore.ParameterSet.Config as cms
import os
import string

from DQM.HcalMonitorModule.HcalMonitorModule_cfi import * # need for setHcalTaskValues, setHcalSubdetTaskValues functions
from DQM.HcalMonitorClient.HcalMonitorClient_cfi import * 

process = cms.Process("HCALDQM")

#------------------------------------------------------
#  variables used in multiple places
#-----------------------------------------------------                      

maxevents      = 1000    # maximum number of events to process
checkNevents   = 1000    # some histograms are filled 'every checkNevents' events; others are filled every luminosity block or every event
debuglevel     = 0      # larger value means more debug messages (0=no debug)
databasedir  = ''       # Set to an existing directory to dump out database info
host = os.getenv("HOST")
host=string.split(host,".")[0]

subsystem="Hcal"        # specify subsystem name  (default is "Hcal")
source = "PoolSource"   # specify source type (PoolSource, NewEventStreamFileReader, HcalTBSource)
memcheck=False          # Dump out memory usage information

#----------------------------
# Specify Event Source
#-----------------------------

#####################  SET THE NUMBER OF EVENTS OVER WHICH TO RUN HERE #################################

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(maxevents)
    )

##################### SET YOUR FILE TO CHECK HERE #####################################################

### Case 1:  PoolSource (from DBS, castor, etc.)

if source=="PoolSource":
    process.source = cms.Source("PoolSource",
                                
                                fileNames = cms.untracked.vstring
                                (
        # Run with ZDC
        '/store/data/BeamCommissioning09/Cosmics/RAW/v1/000/121/993/D04EA868-5FD6-DE11-B372-003048D2BE08.root',

        # Collisions at 2.36 TeV
        #'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/120/F6ADE109-6BE8-DE11-9680-000423D991D4.root',

        # A (relatively) recent run
        #'/store/data/Commissioning09/Calo/RAW/v3/000/118/962/127CDC23-8FC5-DE11-B66D-000423D991D4.root',
        # Calibration triggers only
        #'/store/data/Commissioning09/TestEnables/RAW/v3/000/118/074/84ED101B-03C0-DE11-B33C-000423D94E70.root',
        # cosmics run with known hot cell in HF
        #'/store/data/Commissioning08/Cosmics/RAW/v1/000/067/838/006945C8-40A5-DD11-BD7E-001617DBD556.root',
        #'/store/data/Commissioning08/Cosmics/RAW/v1/000/067/838/FEEE9F50-61A5-DD11-835E-000423D98DD4.root',
        # NON-ZERO-SUPPRESSED RUN
        #'/store/data/Commissioning08/Cosmics/RAW/v1/000/064/103/2A983512-E18F-DD11-BE84-001617E30CA4.root'
        #'/store/data/Commissioning08/Cosmics/RAW/v1/000/066/904/02944F1F-EB9E-DD11-8D88-001D09F2A465.root',
        )
                                )

### Case 2:  Run on raw .dat files

elif source=="NewEventStreamFileReader":
    process.source = cms.Source("NewEventStreamFileReader",
                                fileNames = cms.untracked.vstring(
        #'/store/data/GlobalCruzet3MW33/A/000/056/416/GlobalCruzet3MW33.00056416.0001.A.storageManager.0.0000.dat'
        # example file from online (cmsusr0) directory (lookarea_SM)
        #'file:/lookarea_SM/MWGR_40_2009.00116136.0036.A.storageManager.07.0000.dat',
        #'/store/streamer/RunPrep09/A/000/120/325/RunPrep09.00120325.0002.A.storageManager.06.0001.dat',
        #'/store/streamer/RunPrep09/A/000/120/331/RunPrep09.00120331.0196.A.storageManager.04.0000.dat'
        #Francesco's check
        '/store/streamer/RunPrep09/A/000/120/331/RunPrep09.00120331.0060.A.storageManager.00.0000.dat',
        #'/store/streamer/RunPrep09/A/000/120/331/RunPrep09.00120331.0060.A.storageManager.01.0000.dat',
        '/store/streamer/RunPrep09/A/000/120/331/RunPrep09.00120331.0197.A.storageManager.07.0000.dat'
        )
                                )

### Case 3:  Run on HCAL local runs (pedestal, LED, etc.).  These files are stored on /bigspool/usc/ in cmshcal01, etc.
elif source=="HcalTBSource":
    process.source = cms.Source("HcalTBSource",
                                fileNames = cms.untracked.vstring('file:/afs/cern.ch/user/s/stjohn/scratch0/USC_077834.root'),
                                streams   = cms.untracked.vstring(
        #HBHEa,b,c:
        'HCAL_DCC700','HCAL_DCC701','HCAL_DCC702','HCAL_DCC703','HCAL_DCC704','HCAL_DCC705',
        'HCAL_DCC706','HCAL_DCC707','HCAL_DCC708','HCAL_DCC709','HCAL_DCC710','HCAL_DCC711',
        'HCAL_DCC712','HCAL_DCC713','HCAL_DCC714','HCAL_DCC715','HCAL_DCC716','HCAL_DCC717',
        #HF:
        'HCAL_DCC718','HCAL_DCC719','HCAL_DCC720','HCAL_DCC721','HCAL_DCC722','HCAL_DCC723',
        #HO:
        'HCAL_DCC724','HCAL_DCC725','HCAL_DCC726','HCAL_DCC727','HCAL_DCC728','HCAL_DCC729',
        'HCAL_DCC730','HCAL_DCC731',
        'HCAL_Trigger','HCAL_SlowData'
        )
                                )


#-------------------------------------------
#  Message Logger stuff
#-------------------------------------------

process.load("FWCore.MessageLogger.MessageLogger_cfi")
# Reduce frequency of MessageLogger event output messages
process.MessageLogger.cerr.FwkReport.reportEvery = 100


#----------------------------
# DQM Environment
#-----------------------------
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")

# Set collector host to machine where gui output to be collected
#process.DQM.collectorHost = 'lxplus249'
#process.DQM.collectorHost = host
#process.DQM.collectorPort = 9190
process.dqmSaver.convention = 'Online'
process.dqmSaver.producer = 'DQM'
process.dqmEnv.subSystemFolder = subsystem
# optionally change fileSaving  conditions
# replace dqmSaver.prescaleLS =   -1
# replace dqmSaver.prescaleTime = -1 # in minutes
#process.dqmSaver.prescaleEvt =  2

# For Hcal local run files, replace dqmSaver.saveByRun = 2 
process.dqmSaver.saveByRun = 1


#-----------------------------
# Hcal Conditions: from Global Conditions Tag 
#-----------------------------

# Use https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions
# to choose appropriate global tags

# lxplus
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.globaltag = "GR09_P_V5::All" 
process.GlobalTag.globaltag = "GR10_P_V1::All"
process.es_prefer_GlobalTag = cms.ESPrefer('PoolDBESSource','GlobalTag')
process.prefer("GlobalTag")

# Use these tags when testing in the private network
#process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.connect = "frontier://(proxyurl=http://localhost:3128)(serverurl=http://frontier1.cms:8000/FrontierOnProd)(serverurl=http://frontier2.cms:8000/FrontierOnProd)(retrieve-ziplevel=0)/CMS_COND_31X_GLOBALTAG"
#process.GlobalTag.globaltag = 'GR09_31X_V6H::All' # or any other appropriate
#process.es_prefer_GlobalTag = cms.ESPrefer('PoolDBESSource','GlobalTag')

#-----------------------------
# ValGrind/Igprof Memory Check Information
#------------------------------

# Valgrind -- run with:
'''
 valgrind --tool=memcheck --leak-check=yes --show-reachable=yes \
  `cmsvgsupp` --num-callers=50 --track-fds=yes cmsRun hcal_dqm_sourceclient-file_cfg.py
  '''
# \ >& out.valgrind.txt < /dev/null &


#Igprof -- run with:
'''
igprof -d -t cmsRun -pp -mp -z -o igprof.myprof.gz cmsRun \
hcal_dqm_sourceclient-file_cfg.py > & out.myprof.txt < /dev/null &
'''
#Analyze igprof results with:
'''
igprof-analyse -g -d -v -p -r MEM_MAX igprof.myprof.gz | less
'''

# ENABLE memory check if memcheck bool has been set
if (memcheck):
    process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",
                                            ignoreTotal=cms.untracked.int32(1),
                                            oncePerEventMode=cms.untracked.bool(False)
,                                            )


#---------------------------------------------
# Hcal DQM Source, including HitReconstrctor
#---------------------------------------------

# AT LAST!  Load the DQM HcalMonitorModule!
process.load("DQM.HcalMonitorModule.HcalMonitorModule_cfi")
process.load("DQM.HcalMonitorModule.ZDCMonitorModule_cfi")
process.load("EventFilter.HcalRawToDigi.HcalRawToDigi_cfi")

# This is a separate process to create a special collection of digis in which
# problem digis aren't rejected from the collection.  Casual users do not need to worry about this.
process.hcalAllDigis = cms.EDFilter("HcalRawToDigi",
                                    # Flag to enable unpacking of ZDC channels (default = false)
                                    UnpackZDC = cms.untracked.bool(True),
                                    # Optional filter to remove any digi with "data valid" off, "error" on,
                                    # or capids not rotating
                                    FilterDataQuality = cms.bool(False),
                                    # Do not complain about missing FEDs
                                    ExceptionEmptyData = cms.untracked.bool(False),
                                    InputLabel = cms.InputTag("source"),
                                    # Use the defaults for FED numbers
                                    # Do not complain about missing FEDs
                                    ComplainEmptyData = cms.untracked.bool(False),
                                    # Flag to enable unpacking of calibration channels (default = false)
                                    UnpackCalib = cms.untracked.bool(True),
                                    lastSample = cms.int32(9),
                                    # At most ten samples can be put into a digi, if there are more
                                    # than ten, firstSample and lastSample select which samples
                                    # will be copied to the digi
                                    firstSample = cms.int32(0)
                                    )

process.load("RecoLocalCalo.HcalRecProducers.HcalHitReconstructor_hbhe_cfi")
process.load("RecoLocalCalo.HcalRecProducers.HcalHitReconstructor_ho_cfi")
process.load("RecoLocalCalo.HcalRecProducers.HcalHitReconstructor_hf_cfi")
process.load("RecoLocalCalo.HcalRecProducers.HcalHitReconstructor_zdc_cfi")

# Timing correction to HF reconstruction
process.hfreco.firstSample = 3
process.hfreco.samplesToAdd = 4

# -----------------------------------
# Specify Severity Level Computer
# -----------------------------------

process.essourceSev =  cms.ESSource("EmptyESSource",
                                               recordName = cms.string("HcalSeverityLevelComputerRcd"),
                                               firstValid = cms.vuint32(1),
                                               iovIsRunNotTime = cms.bool(True)
                            )


process.hcalRecAlgos = cms.ESProducer("HcalRecAlgoESProducer",
                                      SeverityLevels = cms.VPSet(
    cms.PSet( Level = cms.int32(0),
              RecHitFlags = cms.vstring(''),
              ChannelStatus = cms.vstring('')
              ),
    cms.PSet( Level = cms.int32(5),
              RecHitFlags = cms.vstring('HSCP_R1R2','HSCP_FracLeader','HSCP_OuterEnergy',
                                        'HSCP_ExpFit','ADCSaturationBit'),
              ChannelStatus = cms.vstring('')
              ),
    cms.PSet( Level = cms.int32(8),
              RecHitFlags = cms.vstring('HBHEHpdHitMultiplicity', 'HBHEPulseShape', 'HOBit',
                                        'HFDigiTime', 'HFLongShort', 'ZDCBit', 'CalibrationBit',
                                        'TimingErrorBit'),
              ChannelStatus = cms.vstring('')
              ),
    cms.PSet( Level = cms.int32(10),
              RecHitFlags = cms.vstring(''),
              ChannelStatus = cms.vstring('HcalCellHot')
              ),
    cms.PSet( Level = cms.int32(20),
              RecHitFlags = cms.vstring(''),
              ChannelStatus = cms.vstring('HcalCellOff', 'HcalCellDead')
              )
    ),
                                      RecoveredRecHitBits = cms.vstring('TimingAddedBit','TimingSubtractedBit'),
                                      DropChannelStatusBits = cms.vstring('HcalCellOff',) #'HcalCellDead' had also been present
                                      )



#----------------------------
# Trigger Emulator
#----------------------------
process.load('SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff')
process.valHcalTriggerPrimitiveDigis = process.simHcalTriggerPrimitiveDigis.clone()
process.valHcalTriggerPrimitiveDigis.inputLabel = cms.VInputTag('hcalDigis', 'hcalDigis')
process.valHcalTriggerPrimitiveDigis.FrontEndFormatError = cms.untracked.bool(True)
process.HcalTPGCoderULUT.LUTGenerationMode = cms.bool(False)


# -------------------------------
# hcalMonitor configurable values
# -------------------------------
process.hcalMonitor.debug = debuglevel
process.hcalMonitor.Online = True # set true for online/local running

process.hcalMonitor.showTiming      = False
process.hcalMonitor.checkNevents    = checkNevents
#process.hcalMonitor.DeadCellMonitor_test_rechits = True

#--------------------------------------------
# Turn on/off individual hcalMonitor modules
#--------------------------------------------
process.hcalMonitor.subSystemFolder = subsystem

process.hcalMonitor.DataFormatMonitor             = True
process.hcalMonitor.DigiMonitor                   = True
process.hcalMonitor.RecHitMonitor                 = True
process.hcalMonitor.TrigPrimMonitor               = True
process.hcalMonitor.DeadCellMonitor               = True
process.hcalMonitor.HotCellMonitor                = True
process.hcalMonitor.BeamMonitor                   = True
process.hcalMonitor.ReferencePedestalMonitor      = True
process.hcalMonitor.LaserMonitor                  = True
process.hcalMonitor.NZSMonitor                    = True

process.hcalMonitor.DetDiagNoiseMonitor           = False
process.hcalMonitor.DetDiagTimingMonitor          = False
process.hcalMonitor.DetDiagLEDMonitor             = False
process.hcalMonitor.DetDiagLaserMonitor           = False
process.hcalMonitor.DetDiagPedestalMonitor        = False

process.hcalMonitor.DataIntegrityTask             = False

# This takes the default cfg values from the hcalMonitor base class and applies them to the subtasks.

setHcalTaskValues(process.hcalMonitor)

process.hcalMonitor.subSystemFolder = subsystem

# Set individual Task values here
#(otherwise they will remain set to the values specified for the hcalMonitor.)

# Loosen HF hot cell thresholds when using cosmic reconstruction
#process.hcalMonitor.HotCellMonitor_HF_energyThreshold = 20
#process.hcalMonitor.HotCellMonitor_HF_persistentThreshold = 10
process.hcalMonitor.HotCellMonitor_AllowedCalibTypes=[0] # ignore calibration (laser, raddam) events
process.hcalMonitor.DeadCellMonitor_LBprescale=1 # set to 4 for online running


# Dump out hflumi information to text file, if directory is specified
process.hcalMonitor.BeamMonitor_lumiqualitydir=""
#---------------------------------------------------------------------
# Hcal DQM Client
#---------------------------------------------------------------------
process.load("DQM.HcalMonitorClient.HcalMonitorClient_cfi")
process.load("DQM.HcalMonitorClient.ZDCMonitorClient_cfi")

# hcalClient configurable values ------------------------
# suppresses html output from HCalClient  
process.hcalClient.baseHtmlDir = ''  # set to '' to ignore html output
process.hcalClient.subSystemFolder  = subsystem

# Set client settings to the same as monitor.
# At the moment, this doesn't affect the client minErrorFlag
# Summary Client is also unaffected, since we want that on all the time

process.hcalClient.databasedir   = databasedir

setHcalClientValuesFromMonitor(process.hcalClient,
                               process.hcalMonitor,
                               debug=debuglevel)

# Keep Summary Client turned on
process.hcalClient.SummaryClient        = True


#----------------------------------------
# Scheduling & Path to follow each event
#-----------------------------------------
process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound', 
        'TooManyProducts', 
        'TooFewProducts')
)

# Allow even bad-quality digis
#process.hcalDigis.FilterDataQuality=False


# Set expected orbit time to 3560 (value should be 6 for run < 116401)
# updated to 3559 in late 2009
process.hcalDigis.ExpectedOrbitMessageTime= cms.untracked.int32(3559)
# Set monitor value to -1 to skip check of IDLE BCN 
process.hcalMonitor.DigiMonitor_ExpectedOrbitMessageTime = 3559

# ----------------------
# Trigger Unpacker Stuff
# ----------------------
process.load("CondCore.DBCommon.CondDBSetup_cfi")
process.load("L1Trigger.Configuration.L1DummyConfig_cff")
process.load("EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi")
process.l1GtUnpack.DaqGtInputTag = 'source'

# -----------------------
# PROCESS
# -----------------------

process.p = cms.Path(process.hcalDigis
                     #*process.hcalAllDigis  # use all digis in digi monitor?
                     *process.valHcalTriggerPrimitiveDigis
                     *process.l1GtUnpack
                     *process.horeco
                     *process.hfreco
                     *process.hbhereco
                     *process.zdcreco
                     *process.hcalMonitor
                     *process.zdcMonitor
                     *process.hcalClient
                     *process.zdcClient
                     *process.dqmEnv
                     *process.dqmSaver
                     )

#-----------------------------
# Quality Tester 
# will add switch to select histograms to be saved soon
#-----------------------------
#process.qTester = cms.EDFilter("QualityTester",
#    prescaleFactor = cms.untracked.int32(1),
#    qtList = cms.untracked.FileInPath('DQM/HcalMonitorClient/data/hcal_qualitytest_config.xml'),
#    getQualityTestsFromFile = cms.untracked.bool(True)
#)

