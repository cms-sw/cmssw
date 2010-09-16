import FWCore.ParameterSet.Config as cms
import os
import string
try:
    import inputfiles
except:
    print "Cannot import file inputfiles.py.  Proceeding anyway..."
process = cms.Process("HCALDQM")

#------------------------------------------------------
#  variables used in multiple places
#-----------------------------------------------------                      

maxevents      = 1000  # maximum number of events to process
debuglevel     = 0     # larger value means more debug messages (0=no debug)
databasedir  = ''       # Set to an existing directory to dump out database info
host = os.getenv("HOST")
if (host.find(".")>-1):
    host=string.split(host,".")[0]
else:
    host=None
user=os.getenv("USER")
htmldir="/tmp/%s"%user
htmldir="" # no html output


subsystem="Hcal"        # specify subsystem name  (default is "Hcal")
source = "PoolSource"   # specify source type (PoolSource, NewEventStreamFileReader, HcalTBSource)
#source="NewEventStreamFileReader"
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
                                # Specify root files to use as inputs here
                                fileNames = cms.untracked.vstring(
        #inputfiles.makelocal(inputfiles.rootfiles,"/tmp/temple/inputfiles")
        inputfiles.rootfiles
        )
                                )

### Case 2:  Run on raw .dat files

elif source=="NewEventStreamFileReader":
    process.source = cms.Source("NewEventStreamFileReader",
                                # Specify .dat files to use here
                                fileNames = cms.untracked.vstring(inputfiles.datfiles)
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
#process.MessageLogger.cerr.FwkReport.reportEvery = 1

#----------------------------
# DQM Environment
#-----------------------------
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")

# Set collector host to machine where gui output to be collected
process.DQM.collectorHost = 'lxplus305'
if (host<>None):
    process.DQM.collectorHost = host
process.DQM.collectorPort = 9190
process.dqmSaver.convention = 'Online'
process.dqmSaver.producer = 'DQM'
process.dqmSaver.dirName='/tmp/%s'%user
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
'''process.hcalMonitorTasksOnlineSequence
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

process.load("EventFilter.HcalRawToDigi.HcalRawToDigi_cfi")

# AT LAST!  Load the DQM HcalMonitorModule!
process.load("DQM.HcalMonitorModule.HcalMonitorModule_cfi")
process.load("DQM.HcalMonitorTasks.HcalMonitorTasks_cfi")
process.load("DQM.HcalMonitorClient.HcalMonitorClient_cfi")

process.hcalBeamMonitor.lumiqualitydir="/tmp/%s/"%user

process.hcalBeamMonitor.skipOutOfOrderLS=False
process.hcalDeadCellMonitor.skipOutOfOrderLS=False
process.hcalDeadCellMonitor.makeDiagnostics=True
process.hcalHotCellMonitor.skipOutOfOrderLS=False
process.hcalHotCellMonitor.test_energy=True
process.hcalHotCellMonitor.test_et=False
process.hcalDigiMonitor.skipOutOfOrderLS=False
process.hcalNZSMonitor.skipOutOfOrderLS=False
process.hcalRecHitMonitor.skipOutOfOrderLS=False
process.hcalTrigPrimMonitor.skipOutOfOrderLS=False

process.hcalClient.debug=0
process.hcalClient.baseHtmlDir=htmldir
process.hcalClient.databaseDir=htmldir
process.hcalClient.minevents=1

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

#----------------------------------------
# Scheduling & Path to follow each event
#-----------------------------------------
process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound', 
        'TooManyProducts', 
        'TooFewProducts')
)

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
                     *process.valHcalTriggerPrimitiveDigis
                     *process.l1GtUnpack
                     *process.horeco
                     *process.hfreco
                     *process.hbhereco
                     *process.zdcreco
                     *process.hcalMonitor
                     *process.hcalMonitorTasksOnlineSequence
                     *process.hcalClient
                     *process.dqmEnv
                     *process.dqmSaver
                     )

