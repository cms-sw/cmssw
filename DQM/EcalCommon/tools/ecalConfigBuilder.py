from optparse import OptionParser
import re

# get options
optparser = OptionParser()
optparser.add_option("-e", "--env", dest = "environment",
    help = "ENV=(CMSLive|PrivLive|PrivOffline|LocalLive|LocalOffline)", metavar = "ENV", default = "")
optparser.add_option("-c", "--config", dest = "config",
    help = "CONFIG=(Physics|Calibration|Laser)", metavar = "CONFIG")
optparser.add_option("-d", "--daqtype", dest = "daqtype", default = "globalDAQ",
    help = "DAQ=(globalDAQ|localDAQ)", metavar = "DAQ")
optparser.add_option("-f", "--file", dest = "filename", default = "",
    help = "write to FILE (optional)", metavar = "FILE")
optparser.add_option("-s", "--source", dest = "sourceFiles", default = "",
    help = "use FILELIST (space separated) as source", metavar = "FILELIST")
optparser.add_option("-g", "--gtag", dest = "gtag", default = "",
    help = "global tag", metavar = "TAG")
optparser.add_option("-w", "--workflow", dest = "workflow", default = "",
    help = "offline workflow", metavar = "WORKFLOW")
optparser.add_option("-r", "--rawdata", dest = "rawdata", default = "",
    help = "collection name", metavar = "RAWDATA")
#optparser.add_option("-t", "--type", dest = "runtype", default = "",
#    help = "ECAL run type", metavar = "RUNTYPE")

(options, args) = optparser.parse_args()

if not options.environment or not options.config :
    optparser.print_usage()
    exit

env = options.environment
if env not in set(['CMSLive', 'PrivLive', 'PrivOffline', 'LocalLive', 'LocalOffline']) :
    optparser.error("ENV value " + env + " not correct")
    exit

config = options.config
if config not in set(['Physics', 'Calibration', 'Laser']) :
    optparser.error("CONFIG value " + config + " not correct")
    exit

daqtype = options.daqtype
if daqtype not in set(['localDAQ', 'globalDAQ', 'miniDAQ']) :
    optparser.error("DAQ value " + daqtype + " not correct")
    exit

#runtype = options.runtype
#if runtype not '' and daqtype not 'localDAQ':
#    optparser.error("Run type can only be set in localDAQ runs")
#    exit

filename = options.filename
sourceFiles = re.sub(r'([^ ]+)[ ]?', r'    "file:\1",\n', options.sourceFiles)
gtag = options.gtag
workflow = options.workflow
FedRawData = options.rawdata

physics = (config == 'Physics')
calib = (config == 'Calibration')
laser = (config == 'Laser')

# set environments

# TODO : should we write in PrivOffline rather than PrivLive?
# PrivLive will have larger coverage of time; what about statistics?

subsystem = 'Ecal'
if laser or calib :
    subsystem = 'EcalCalibration'

central = False
privCentral = False
if (env == 'CMSLive') :
    if laser :
        privCentral = True
    else :
        central = True

privEcal = privCentral
if (env == 'PrivLive') or (env == 'PrivOffline') :
    privEcal = True

p5 = privEcal or central

local = False
if (env == 'LocalOffline') or (env == 'LocalLive') :
    local = True

doOutput = True
if laser and privEcal :
    doOutput = False

live = False
if (env == 'CMSLive') or (env == 'PrivLive') or (env == 'LocalLive') :
    live = True

if not p5 and not gtag :
    optparser.error("Global tag needed for non-P5 DQM cfg")
    exit

if not live and not sourceFiles :
    optparser.error("Source file name not given for offline DQM")
    exit

if doOutput and not live and not workflow :
    optparser.error("Workflow needed for offline DQM")
    exit

if workflow and not re.match('[\/][a-zA-Z0-9_]+[\/][a-zA-Z0-9_]+[\/][a-zA-Z0-9_]+', workflow) :
    optparser.error("Invalid workflow: " + workflow)
    exit

streamerInput = False
if re.search('[.]dat', sourceFiles) :
    streamerInput = True

header = ''
recoModules = ''
ecalDQM = ''
dqmModules = ''
filters = ''
setup = ''
misc = ''
sequencePaths = ''
source = ''
customizations = ''

header += '''import FWCore.ParameterSet.Config as cms
import os, sys, socket

process = cms.Process("DQM")
'''

recoModules += '''

### RECONSTRUCTION MODULES ###

process.load("Geometry.CaloEventSetup.CaloGeometry_cfi")

process.load("Geometry.CaloEventSetup.CaloTopology_cfi")

process.load("Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi")

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.load("Geometry.EcalMapping.EcalMapping_cfi")

process.load("Geometry.EcalMapping.EcalMappingRecord_cfi")
'''


if not laser :
    recoModules += '''
from EventFilter.EcalRawToDigi.EcalUnpackerData_cfi import ecalEBunpacker
process.ecalDigis = ecalEBunpacker.clone()

process.load("RecoLocalCalo.EcalRecProducers.ecalGlobalUncalibRecHit_cfi")

process.load("RecoLocalCalo.EcalRecProducers.ecalDetIdToBeRecovered_cfi")

process.load("RecoLocalCalo.EcalRecProducers.ecalRecHit_cfi")

process.load("RecoLocalCalo.EcalRecAlgos.EcalSeverityLevelESProducer_cfi")

process.load("RecoEcal.EgammaClusterProducers.ecalClusteringSequence_cff")

process.load("CalibCalorimetry.EcalLaserCorrection.ecalLaserCorrectionService_cfi")

process.load("SimCalorimetry.EcalTrigPrimProducers.ecalTriggerPrimitiveDigis_cfi")
'''

if calib :
    recoModules += '''
from RecoLocalCalo.EcalRecProducers.ecalFixedAlphaBetaFitUncalibRecHit_cfi import ecalFixedAlphaBetaFitUncalibRecHit
process.ecalLaserLedUncalibRecHit = ecalFixedAlphaBetaFitUncalibRecHit.clone()

from RecoLocalCalo.EcalRecProducers.ecalMaxSampleUncalibRecHit_cfi import ecalMaxSampleUncalibRecHit
process.ecalTestPulseUncalibRecHit = ecalMaxSampleUncalibRecHit.clone()
'''


ecalDQM += '''
    
### ECAL DQM MODULES ###

process.load("DQM.EcalCommon.EcalDQMBinningService_cfi")
'''
if physics:
    ecalDQM += '''
process.load("DQM.EcalBarrelMonitorTasks.EcalMonitorTask_cfi")
process.load("DQM.EcalBarrelMonitorClient.EcalMonitorClient_cfi")
'''

if calib :
    ecalDQM += '''
process.load("DQM.EcalBarrelMonitorTasks.EcalCalibMonitorTasks_cfi")
process.load("DQM.EcalBarrelMonitorClient.EcalCalibMonitorClient_cfi")
'''        
    if daqtype == 'localDAQ':
        ecalDQM += '''
process.load("DQM.EcalBarrelMonitorTasks.EcalMonitorTask_cfi")

process.ecalMonitorTask.workers = ["IntegrityTask", "RawDataTask"]
process.ecalCalibMonitorClient.workers = ["IntegrityClient", "RawDataClient", "LaserClient", "LedClient", "TestPulseClient", "PedestalClient", "PNIntegrityClient", "SummaryClient", "CalibrationSummaryClient"]
process.ecalCalibMonitorClient.workerParameters.SummaryClient.activeSources = ["Integrity", "RawData"]
'''
    #Need to comfigure the source for calib summary!!

if Ecal and not live:
    ecalDQM += '''
process.load("Toolset.DQMTools.DQMFileLoader_cfi")
'''

dqmModules += '''

### DQM COMMON MODULES ###

process.load("DQMServices.Core.DQM_cfg")
'''

if physics :
    dqmModules += '''
process.dqmQTest = cms.EDAnalyzer("QualityTester",
    reportThreshold = cms.untracked.string("red"),
    prescaleFactor = cms.untracked.int32(1),
    qtList = cms.untracked.FileInPath("DQM/EcalCommon/data/EcalQualityTests.xml"),
    getQualityTestsFromFile = cms.untracked.bool(True),
    qtestOnEndLumi = cms.untracked.bool(True),
    qtestOnEndRun = cms.untracked.bool(True)
)
'''

if p5 :
    dqmModules += '''
process.load("DQM.Integration.test.environment_cfi")
'''
else :
    dqmModules += '''
process.load("DQMServices.Components.DQMEnvironment_cfi")
'''


filters += '''

### FILTERS ###

process.load("FWCore.Modules.preScaler_cfi")
'''

if physics :
    filters += '''
process.ecalPhysicsFilter = cms.EDFilter("EcalMonitorPrescaler")
'''
elif calib :
    filters += '''
process.ecalCalibrationFilter = cms.EDFilter("EcalMonitorPrescaler")
process.ecalLaserLedFilter = cms.EDFilter("EcalMonitorPrescaler")
process.ecalTestPulseFilter = cms.EDFilter("EcalMonitorPrescaler")'''
    if (daqtype == 'localDAQ') :
        filters += '''
process.ecalPedestalFilter = cms.EDFilter("EcalMonitorPrescaler")'''


setup += '''


### JOB PARAMETERS ###

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(-1)
)
'''

frontier = 'frontier://FrontierProd/'
if p5 :
    frontier = 'frontier://(proxyurl=http://localhost:3128)(serverurl=http://localhost:8000/FrontierOnProd)(serverurl=http://localhost:8000/FrontierOnProd)(retrieve-ziplevel=0)/'

if not gtag :
    setup += '''
process.load("DQM.Integration.test.FrontierCondition_GT_cfi")
'''
else :
    setup += 'process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")' + "\n"
    setup += 'process.GlobalTag.globaltag = "' + gtag + '::All"' + "\n"

setup += '''
process.GlobalTag.toGet = cms.VPSet(
    cms.PSet(
        record = cms.string("EcalDQMChannelStatusRcd"),
        tag = cms.string("EcalDQMChannelStatus_v1_hlt"),
        connect = cms.untracked.string("''' + frontier + '''CMS_COND_34X_ECAL")
    ),
    cms.PSet(
        record = cms.string("EcalDQMTowerStatusRcd"),
        tag = cms.string("EcalDQMTowerStatus_v1_hlt"),
        connect = cms.untracked.string("''' + frontier + '''CMS_COND_34X_ECAL")
    )
)
'''


misc += '''

### MESSAGE LOGGER ###

process.MessageLogger = cms.Service("MessageLogger",
  cout = cms.untracked.PSet(
    threshold = cms.untracked.string("WARNING"),
    noLineBreaks = cms.untracked.bool(True),
    noTimeStamps = cms.untracked.bool(True),
    default = cms.untracked.PSet(
      limit = cms.untracked.int32(0)
    )
  ),
  destinations = cms.untracked.vstring("cout")
)
'''


sequencePaths += '''

### SEQUENCES AND PATHS ###
'''

if laser :
    sequencePaths += '''
process.ecalMonitorPath = cms.Path(
    process.ecalLaserMonitorClient
)
'''
else :
    sequencePaths += '''
process.ecalPreRecoSequence = cms.Sequence(
    process.preScaler +
    process.ecalDigis
)

process.ecalRecoSequence = cms.Sequence(
    process.ecalGlobalUncalibRecHit *
    process.ecalDetIdToBeRecovered *
    process.ecalRecHit
)
'''

if physics :
    sequencePaths += '''
process.ecalClusterSequence = cms.Sequence(
    process.hybridClusteringSequence *
    process.multi5x5ClusteringSequence
)
process.ecalClusterSequence.remove(process.multi5x5SuperClustersWithPreshower)

process.ecalMonitorPath = cms.Path(
    process.ecalPreRecoSequence *
    process.ecalPhysicsFilter *
    process.ecalRecoSequence *
    process.ecalClusterSequence *
    process.simEcalTriggerPrimitiveDigis *
    process.ecalMonitorTask
)
'''

if calib :
    sequencePaths += '''
process.ecalLaserLedPath = cms.Path(
    process.ecalPreRecoSequence *
    process.ecalLaserLedFilter *    
    process.ecalRecoSequence *
    process.ecalLaserLedUncalibRecHit *
    process.ecalLaserLedMonitorTask *
    process.ecalPNDiodeMonitorTask
)
process.ecalTestPulsePath = cms.Path(
    process.ecalPreRecoSequence *
    process.ecalTestPulseFilter *    
    process.ecalRecoSequence *
    process.ecalTestPulseUncalibRecHit *
    process.ecalTestPulseMonitorTask *
    process.ecalPNDiodeMonitorTask
)
'''
    if (daqtype == 'localDAQ') :
        sequencePaths += '''
process.ecalPedestalPath = cms.Path(
    process.ecalPreRecoSequence *
    process.ecalPedestalFilter *    
    process.ecalRecoSequence *
    process.ecalPedestalMonitorTask *
    process.ecalPNDiodeMonitorTask
)
process.ecalMonitorPath = cms.Path(
    process.ecalPreRecoSequence *
    process.ecalMonitorTask
)
'''


sequencePaths += '''
process.ecalClientPath = cms.Path('''
if physics :
    sequencePaths += '''
    process.ecalMonitorClient'''        
else :
    sequencePaths += '''
    process.ecalPreRecoSequence *
    process.ecalCalibrationFilter *
    process.ecalCalibMonitorClient'''

sequencePaths += '''
)

process.dqmEndPath = cms.EndPath(
    process.dqmEnv'''

if physics :
    sequencePaths += ''' *
    process.dqmQTest'''
    
if doOutput :
    if privEcal and live:
        sequencePaths += ''' *
    process.dqmFileLoader'''
        
    sequencePaths += ''' *
    process.dqmSaver'''

sequencePaths += '''
)

process.schedule = cms.Schedule('''

if physics :
    sequencePaths += '''
    process.ecalMonitorPath,
    process.ecalClientPath,'''

if calib :
    sequencePaths += '''
    process.ecalLaserLedPath,
    process.ecalTestPulsePath,
    process.ecalClientPath,'''

    if (daqtype == 'localDAQ') :
        sequencePaths += '''
    process.ecalPedestalPath,
    process.ecalMonitorPath,'''
    
elif laser :
    sequencePaths += '''
    process.ecalMonitorPath,'''

sequencePaths += '''
    process.dqmEndPath
)
'''


source += '''

### SOURCE ###
'''

if live :
    source += '''
process.load("DQM.Integration.test.inputsource_cfi")
'''
else :
    if streamerInput :
        source += '''
process.source = cms.Source("NewEventStreamFileReader")
'''
    else :
        source += '''
process.source = cms.Source("PoolSource")
'''


customizations += '''

### CUSTOMIZATIONS ###
'''

if not laser :
    customizations += '''
 ## Reconstruction Modules ##

process.ecalRecHit.killDeadChannels = True
process.ecalRecHit.ChannelStatusToBeExcluded = [ 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 78, 142 ]
'''

if physics :
    customizations += '''
process.simEcalTriggerPrimitiveDigis.Label = "ecalDigis"
process.simEcalTriggerPrimitiveDigis.InstanceEB = "ebDigis"
process.simEcalTriggerPrimitiveDigis.InstanceEE = "eeDigis"
'''
elif calib :
    customizations += '''
process.ecalTestPulseUncalibRecHit.EBdigiCollection = "ecalDigis:ebDigis"
process.ecalTestPulseUncalibRecHit.EEdigiCollection = "ecalDigis:eeDigis"
    
process.ecalLaserLedUncalibRecHit.MinAmplBarrel = 12.
process.ecalLaserLedUncalibRecHit.MinAmplEndcap = 16.
'''

customizations += '''
 ## Filters ##
'''

if physics :
    customizations += '''
process.ecalPhysicsFilter.EcalRawDataCollection = cms.InputTag("ecalDigis")
process.ecalPhysicsFilter.clusterPrescaleFactor = cms.untracked.int32(1)
'''
elif calib :
    customizations += '''
process.ecalCalibrationFilter.EcalRawDataCollection = cms.InputTag("ecalDigis")
process.ecalCalibrationFilter.laserPrescaleFactor = cms.untracked.int32(1)
process.ecalCalibrationFilter.ledPrescaleFactor = cms.untracked.int32(1)
process.ecalCalibrationFilter.pedestalPrescaleFactor = cms.untracked.int32(1)
process.ecalCalibrationFilter.testpulsePrescaleFactor = cms.untracked.int32(1)

process.ecalLaserLedFilter.EcalRawDataCollection = cms.InputTag("ecalDigis")
process.ecalLaserLedFilter.laserPrescaleFactor = cms.untracked.int32(1)
process.ecalLaserLedFilter.ledPrescaleFactor = cms.untracked.int32(1)

process.ecalTestPulseFilter.EcalRawDataCollection = cms.InputTag("ecalDigis")
process.ecalTestPulseFilter.testpulsePrescaleFactor = cms.untracked.int32(1)
'''
    if (daqtype == 'localDAQ') :
        customizations += '''
process.ecalPedestalFilter.EcalRawDataCollection = cms.InputTag("ecalDigis")
process.ecalPedestalFilter.pedestalPrescaleFactor = cms.untracked.int32(1)
'''

customizations += '''
 ## Ecal DQM modules ##
'''

if physics :
    customizations += '''
process.ecalMonitorTask.online = True
process.ecalMonitorTask.workers = ["ClusterTask", "EnergyTask", "IntegrityTask", "OccupancyTask", "RawDataTask", "TimingTask", "TrigPrimTask", "PresampleTask", "SelectiveReadoutTask"]
process.ecalMonitorTask.workerParameters.common.hltTaskMode = 0
process.ecalMonitorTask.workerParameters.TrigPrimTask.runOnEmul = True

process.ecalMonitorClient.online = True
process.ecalMonitorClient.workers = ["IntegrityClient", "OccupancyClient", "PresampleClient", "RawDataClient", "TimingClient", "SelectiveReadoutClient", "TrigPrimClient", "SummaryClient"]
process.ecalMonitorClient.workerParameters.SummaryClient.activeSources = ["Integrity", "RawData", "Presample", "TriggerPrimitives", "Timing", "HotCell"]
'''
elif laser :
    if privEcal :
        customizations += '''
process.ecalLaserMonitorClient.clientParameters.LightChecker.matacqPlotsDir = "/data/ecalod-disk01/dqm-data/laser"
'''

if daqtype == 'localDAQ':
    customizations += '''
process.ecalPedestalMonitorTask.workerParameters.common.MGPAGains = [1, 6, 12]
process.ecalPedestalMonitorTask.workerParameters.common.MGPAGainsPN = [1, 16]
process.ecalTestPulseMonitorTask.workerParameters.common.MGPAGains = [1, 6, 12]
process.ecalTestPulseMonitorTask.workerParameters.common.MGPAGainsPN = [1, 16]
process.ecalPNDiodeMonitorTask.workerParameters.common.MGPAGainsPN = [1, 16]
process.ecalCalibMonitorClient.workerParameters.common.MGPAGains = [1, 6, 12]
process.ecalCalibMonitorClient.workerParameters.common.MGPAGainsPN = [1, 16]
'''

if privEcal and not live:
    customizations += '''
process.dqmFileLoader.directory = "/data/ecalod-disk01/dqm-data/tmp"
'''

customizations += '''
 ## DQM common modules ##
'''

if laser or calib:
    customizations += '''
process.dqmEnv.subSystemFolder = cms.untracked.string("EcalCalibration")
'''
else :
    customizations += '''
process.dqmEnv.subSystemFolder = cms.untracked.string("Ecal")
'''

if central :
    if physics :
        customizations += '''
process.DQMStore.referenceFileName = "/dqmdata/dqm/reference/ecal_reference.root"
'''
    elif calib :
        customizations += '''
process.DQMStore.referenceFileName = "/dqmdata/dqm/reference/ecalcalib_reference.root"
'''

if doOutput :
    if not central or privCentral :
        customizations += 'process.dqmSaver.referenceHandling = "skip"' + "\n"

    if privEcal :
        customizations += '''
process.dqmSaver.saveByTime = -1
process.dqmSaver.saveByMinute = -1'''
        if live :
            customizations += '''
process.dqmSaver.convention = "Online"
process.dqmSaver.dirName = "/data/ecalod-disk01/dqm-data/online-DQM/data"
'''
        else :
            customizations += '''
process.dqmSaver.dirName = "/data/ecalod-disk01/dqm-data/tmp"
'''

    if not live :
        customizations += 'process.dqmSaver.convention = "Offline"' + "\n"
        customizations += 'process.dqmSaver.workflow = "' + workflow + '"' + "\n"

if local :
    customizations += '''
process.DQM.collectorHost = "localhost"
process.DQM.collectorPort = 8061
'''
elif live and privEcal and not privCentral :
        customizations += '''
process.DQM.collectorHost = "ecalod-web01.cms"
process.DQM.collectorPort = 9190
'''
elif not central and not privCentral :
    customizations += '''
process.DQM.collectorHost = ""
'''

customizations += '''
 ## Source ##
'''
if live :
    customizations += 'process.source.consumerName = cms.untracked.string("' + subsystem + ' DQM Consumer")' + "\n"
    if privEcal :
        customizations += 'process.source.sourceURL = cms.string("http://dqm-c2d07-30.cms:22100/urn:xdaq-application:lid=30")' + "\n"
    elif local :
        customizations += 'process.source.sourceURL = cms.string("http://localhost:22100/urn:xdaq-application:lid=30")' + "\n"

    if not physics or (daqtype != 'globalDAQ') :
        customizations += 'process.source.SelectHLTOutput = cms.untracked.string("hltOutputCalibration")' + "\n"
        customizations += 'process.source.SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring("HLT_EcalCalibration_v*"))' + "\n"
        
else :
    customizations += '''
process.source.fileNames = cms.untracked.vstring(
''' + sourceFiles + '''
)
'''

if p5 :
    customizations += '''
 ## Run type specific ##
'''
    if physics : 
        customizations += '''
if process.runType.getRunType() == process.runType.cosmic_run :
    process.dqmEndPath.remove(process.dqmQTest)
    process.ecalMonitorTask.workers = ["EnergyTask", "IntegrityTask", "OccupancyTask", "RawDataTask", "TrigPrimTask", "PresampleTask", "SelectiveReadoutTask"]
    process.ecalMonitorClient.workers = ["IntegrityClient", "OccupancyClient", "PresampleClient", "RawDataClient", "SelectiveReadoutClient", "TrigPrimClient", "SummaryClient"]
    process.ecalMonitorClient.workerParameters.SummaryClient.activeSources = ["Integrity", "RawData", "Presample", "TriggerPrimitives", "HotCell"]
elif process.runType.getRunType() == process.runType.hpu_run:
    process.source.SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring("*"))
'''

HIFedRawData = 'rawDataRepacker'
if (FedRawData == '') :
    if not physics and (daqtype == 'globalDAQ') :
        FedRawData = 'hltEcalCalibrationRaw'
    else :
        FedRawData = 'rawDataCollector'
    
customizations += '''
 ## FEDRawDataCollection name ##
FedRawData = "''' + FedRawData + '''"
'''

if p5 and physics :
    customizations += '''
if process.runType.getRunType() == process.runType.hi_run:
    FedRawData = "''' + HIFedRawData + '''"
'''

if not laser :
    customizations += '''
process.ecalDigis.InputLabel = cms.InputTag(FedRawData)
'''

if physics :
    customizations += 'process.ecalMonitorTask.collectionTags.Source = FedRawData' + "\n"

# write cfg file
if filename == '' :
    if physics :
        c = 'ecal'
    elif calib :
        c = 'ecalcalib'
    elif laser :
        c = 'ecallaser'

    if central or privCentral :
        e = 'live'
    elif privEcal and live :
        e = 'privlive'
    elif live :
        e = 'locallive'
    else :
        e = 'data'

    filename = c + '_dqm_sourceclient-' + e + '_cfg.py'

cfgfile = file(filename, "w")

cfgfile.write(
    header +
    recoModules +
    ecalDQM +
    dqmModules +
    filters +
    setup +
    misc +
    sequencePaths +
    source +
    customizations
)

cfgfile.close()
