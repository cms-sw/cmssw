import FWCore.ParameterSet.Config as cms
from DQM.HcalMonitorModule.HcalMonitorModule_cfi import * # there's probably a better way to do this, once I discover the difference between import and load
from DQM.HcalMonitorClient.HcalMonitorClient_cfi import * # ditto


process = cms.Process("HCALDQM")

#-----------------------------------
#  variables used in multiple places
#-----------------------------------                      

maxevents=1000          # maximum number of events to process
checkNevents=1000       # histograms are filled 'every checkNevents' events
subsystem="Hcal"        # specify subsystem name  (default is "Hcal")
source = "PoolSource"   # specify source type (PoolSource, NewEventStreamFileReader, HcalTBSource)
#source="NewEventStreamFileReader"
memcheck=False

process.load("FWCore.MessageLogger.MessageLogger_cfi")
# Tone down the logging messages, MessageLogger!
process.MessageLogger.cerr.FwkReport.reportEvery = 500

#----------------------------
# Event Source
#-----------------------------

#####################  SET THE NUMBER OF EVENTS OVER WHICH TO RUN HERE #################################

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(maxevents)
    )

##################### SET YOUR FILE TO CHECK HERE #####################################################

### Case 1:  PoolSource (from DBS, etc.)

if source=="PoolSource":
    process.source = cms.Source("PoolSource",
                                
                                fileNames = cms.untracked.vstring
                                (
        # CRAFT 09 ZS run
        '/store/data/CRAFT09/Calo/RAW/v1/000/110/972/FEA7E7DF-E788-DE11-8BAB-001617E30CC8.root',
        # cosmics run with known hot cell in HF
        #'/store/data/Commissioning08/Cosmics/RAW/v1/000/067/838/006945C8-40A5-DD11-BD7E-001617DBD556.root',
        #'/store/data/Commissioning08/Cosmics/RAW/v1/000/067/838/FEEE9F50-61A5-DD11-835E-000423D98DD4.root',
        # NON-ZERO-SUPPRESSED RUN
        #'/store/data/Commissioning08/Cosmics/RAW/v1/000/064/103/2A983512-E18F-DD11-BE84-001617E30CA4.root'
        #'/store/data/Commissioning08/Cosmics/RAW/v1/000/066/904/02944F1F-EB9E-DD11-8D88-001D09F2A465.root',
        # ZERO-SUPPRESSED RUN
        #'/store/data/Commissioning08/Cosmics/RAW/v1/000/064/042/0A36AA7D-978F-DD11-BA36-000423D6C8E6.root'
        #'/store/data/Commissioning08/Cosmics/RAW/v1/000/065/675/5A40BDBD-0399-DD11-88A4-001617E30CE8.root'
        )
                                )

elif source=="NewEventStreamFileReader":
    #Case 2:  Run on raw .dat files
    process.source = cms.Source("NewEventStreamFileReader",
                                fileNames = cms.untracked.vstring(
        #'/store/data/GlobalCruzet3MW33/A/000/056/416/GlobalCruzet3MW33.00056416.0001.A.storageManager.0.0000.dat'
        'file:/lookarea_SM/MWGR_40_2009.00116136.0036.A.storageManager.07.0000.dat',
        )
                                )


elif source=="HcalTBSource":
    #  Case 3:  Run on HCAL local runs (pedestal, LED, etc.).  You need to have access to /bigspool or have a local copy of the file available.

    process.source = cms.Source("HcalTBSource",
                                fileNames = cms.untracked.vstring('file:/afs/cern.ch/user/s/stjohn/scratch0/USC_077834.root'),
                                streams   = cms.untracked.vstring(#HBHEa,b,c:
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

#----------------------------
# DQM Environment
#-----------------------------
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")

process.DQM.collectorHost = 'cmsru2'
process.DQM.collectorPort = 9190
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

# lxplus
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "GR09_31X_V6P::All" # tags listed at SWGuideFrontierConditions twiki
process.es_prefer_GlobalTag = cms.ESPrefer('PoolDBESSource','GlobalTag')
process.prefer("GlobalTag")

# online testing
#process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.connect = "frontier://(proxyurl=http://localhost:3128)(serverurl=http://frontier1.cms:8000/FrontierOnProd)(serverurl=http://frontier2.cms:8000/FrontierOnProd)(retrieve-ziplevel=0)/CMS_COND_31X_GLOBALTAG"
#process.GlobalTag.globaltag = 'GR09_31X_V6H::All' # or any other appropriate
#process.es_prefer_GlobalTag = cms.ESPrefer('PoolDBESSource','GlobalTag')

#----------------------
# ValGrind/Igprof Memory Check
#----------------------

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

if (memcheck):
    process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",
                                            ignoreTotal=cms.untracked.int32(1),
                                            oncePerEventMode=cms.untracked.bool(False)
,                                            )


#-----------------------------
# Hcal DQM Source, including HitReconstrctor
#-----------------------------
process.load("DQM.HcalMonitorModule.HcalMonitorModule_cfi")
process.load("EventFilter.HcalRawToDigi.HcalRawToDigi_cfi")

# Make separate process to look at all digis?
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

# Cosmics Corrections to reconstruction
process.hbhereco.firstSample = 1
process.hbhereco.samplesToAdd = 8
process.hbhereco.correctForTimeslew = True
process.hbhereco.correctForPhaseContainment = True
process.hbhereco.correctionPhaseNS = 10.0
process.horeco.firstSample = 1
process.horeco.samplesToAdd = 8
process.horeco.correctForTimeslew = True
process.horeco.correctForPhaseContainment = True
process.horeco.correctionPhaseNS = 10.
process.hfreco.firstSample = 1
process.hfreco.samplesToAdd = 8
process.hfreco.correctForTimeslew = True
process.hfreco.correctForPhaseContainment = True
process.hfreco.correctionPhaseNS = 10.
process.zdcreco.firstSample = 1
process.zdcreco.samplesToAdd = 8
process.zdcreco.correctForTimeslew = True
process.zdcreco.correctForPhaseContainment = True
process.zdcreco.correctionPhaseNS = 10.


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
# Emulator
#----------------------------
process.load('SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff')
process.valHcalTriggerPrimitiveDigis = process.simHcalTriggerPrimitiveDigis.clone()
process.valHcalTriggerPrimitiveDigis.inputLabel = cms.VInputTag('hcalDigis', 'hcalDigis')
process.HcalTPGCoderULUT.LUTGenerationMode = cms.bool(False)



# -------------------------------
# hcalMonitor configurable values
# ------- -----------------------
process.hcalMonitor.debug = 0
process.hcalMonitor.Online = True # set true for online/local running

process.hcalMonitor.showTiming      = False
process.hcalMonitor.checkNevents    = checkNevents
process.hcalMonitor.dump2database   = False
process.hcalMonitor.AnalyzeOrbitGap = False

#--------------------------------------------
# Turn on/off individual hcalMonitor modules
#--------------------------------------------
process.hcalMonitor.subSystemFolder = subsystem

process.hcalMonitor.DataFormatMonitor   = True
process.hcalMonitor.DataIntegrityTask   = True
process.hcalMonitor.DigiMonitor         = True
process.hcalMonitor.RecHitMonitor       = True
process.hcalMonitor.TrigPrimMonitor     = True
process.hcalMonitor.DeadCellMonitor     = True
process.hcalMonitor.HotCellMonitor      = True
process.hcalMonitor.BeamMonitor         = True
process.hcalMonitor.PedestalMonitor     = True
process.hcalMonitor.DetDiagNoiseMonitor = True
process.hcalMonitor.DetDiagTimingMonitor = True
process.hcalMonitor.LEDMonitor          = False
process.hcalMonitor.CaloTowerMonitor    = False
process.hcalMonitor.MTCCMonitor         = False
process.hcalMonitor.HcalAnalysis        = False

# This takes the default cfg values from the hcalMonitor base class and applies them to the subtasks.
setHcalTaskValues(process.hcalMonitor)

# values are normally 1000, 10
# ProblemCells evaluated every (DeadCellMonitor_checkNevents)
# occupancy/energy tests performed every (DeadCellMonitor_checkNevents*prescale)
process.hcalMonitor.DeadCellMonitor_checkNevents = checkNevents
process.hcalMonitor.DeadCellMonitor_prescale = 10

process.hcalMonitor.subSystemFolder = subsystem

# Set individual Task values here (otherwise they will remain set to the values specified for the hcalMonitor.)

# Loosen HF hot cell thresholds when using cosmic reconstruction
hcalMonitor.HotCellMonitor_HF_energyThreshold = 20
hcalMonitor.HotCellMonitor_HF_persistentThreshold = 10

#-----------------------------
# Hcal DQM Client
#-----------------------------
process.load("DQM.HcalMonitorClient.HcalMonitorClient_cfi")

# hcalClient configurable values ------------------------
# suppresses html output from HCalClient  
process.hcalClient.baseHtmlDir = ''  # set to '' to ignore html output
process.hcalClient.subSystemFolder  = subsystem
#Unnecessary?
#process.hcalClient.prefixME = cms.untracked.string(subsystem)

# Set client settings to the same as monitor.  At the moment, this doesn't affect client minErrorFlag
# Summary Client is also unaffected
setHcalClientValuesFromMonitor(process.hcalClient,process.hcalMonitor, debug=False)  # turn debug to True to dump out client settings

process.hcalClient.SummaryClient        = True

#-----------------------------
# Scheduling & Path to follow each event
#-----------------------------
process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound', 
        'TooManyProducts', 
        'TooFewProducts')
)

# Allow even bad-quality digis
#process.hcalDigis.FilterDataQuality=False

# Set expected orbit time to 3563 (value should be 6 for run < 116401)
process.hcalDigis.ExpectedOrbitMessageTime= 3563
# Set monitor value to -1 to skip check of IDLE BCN 
process.hcalMonitor.DigiMonitor_ExpectedOrbitMessageTime = 3563

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
                     *process.hcalClient
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

