from optparse import OptionParser
import re

# get options
optparser = OptionParser()
optparser.add_option("-e", "--env", dest = "environment",
                   help = "ENV=(CMSLive|PrivLive|PrivOffline|LocalLive|LocalOffline)", metavar = "ENV")
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
optparser.add_option("-n", "--new", action = "store_true", dest = "newFramework")

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

filename = options.filename
sourceFiles = re.sub(r'([^ ]+)[ ]?', r'    "file:\1",\n', options.sourceFiles)
gtag = options.gtag
workflow = options.workflow
FedRawData = options.rawdata
newFramework = options.newFramework

physics = (config == 'Physics')
calib = (config == 'Calibration')
laser = (config == 'Laser')

# set environments

# TODO : should we write in PrivOffline rather than PrivLive?
# PrivLive will have larger coverage of time; what about statistics?

workspace = 'Ecal'
if config == 'Laser' :
    workspace = 'EcalCalibration'

withDB = False
if (env == 'CMSLive') or (env == 'PrivLive') or (daqtype == 'localDAQ') :
    withDB = True

if laser :
    newFramework = True
    
if newFramework :
    withDB = False

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
if (env == 'LocalLive') or (laser and privEcal) :
    doOutput = False

live = False
if (env == 'CMSLive') or (env == 'PrivLive') or (env == 'LocalLive') :
    live = True

if not p5 and not gtag :
    optparser.error("Global tag must be given for non-P5 DQM cfg")
    exit

if not live and not sourceFiles :
    optparser.error("Source file name not given for offline DQM")
    exit

if doOutput and not live and not workflow :
    optparser.error("Workflow needs to be given for offline DQM")
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

process.load("EventFilter.EcalRawToDigi.EcalUnpackerMapping_cfi")

process.load("Geometry.CaloEventSetup.CaloGeometry_cfi")

process.load("Geometry.CaloEventSetup.CaloTopology_cfi")

process.load("Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi")

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.load("Geometry.EcalMapping.EcalMapping_cfi")

process.load("Geometry.EcalMapping.EcalMappingRecord_cfi")
'''

if not laser :
    recoModules += '''
process.load("EventFilter.EcalRawToDigi.EcalUnpackerData_cfi")

import RecoLocalCalo.EcalRecProducers.ecalGlobalUncalibRecHit_cfi
process.ecalUncalibHit = RecoLocalCalo.EcalRecProducers.ecalGlobalUncalibRecHit_cfi.ecalGlobalUncalibRecHit.clone()

process.load("RecoLocalCalo.EcalRecProducers.ecalDetIdToBeRecovered_cfi")

process.load("RecoLocalCalo.EcalRecProducers.ecalRecHit_cfi")

process.load("RecoLocalCalo.EcalRecAlgos.EcalSeverityLevelESProducer_cfi")

process.load("RecoEcal.EgammaClusterProducers.ecalClusteringSequence_cff")

process.load("CalibCalorimetry.EcalLaserCorrection.ecalLaserCorrectionService_cfi")

process.load("SimCalorimetry.EcalTrigPrimProducers.ecalTriggerPrimitiveDigis_cfi")
'''

if physics and live :
    recoModules += '''
process.load("EventFilter.L1GlobalTriggerRawToDigi.l1GtEvmUnpack_cfi")
'''

if calib :
    recoModules += '''
import RecoLocalCalo.EcalRecProducers.ecalFixedAlphaBetaFitUncalibRecHit_cfi
process.ecalUncalibHit1 = RecoLocalCalo.EcalRecProducers.ecalFixedAlphaBetaFitUncalibRecHit_cfi.ecalFixedAlphaBetaFitUncalibRecHit.clone()

import RecoLocalCalo.EcalRecProducers.ecalMaxSampleUncalibRecHit_cfi
process.ecalUncalibHit2 = RecoLocalCalo.EcalRecProducers.ecalMaxSampleUncalibRecHit_cfi.ecalMaxSampleUncalibRecHit.clone()
'''


if newFramework :
    ecalDQM += '''
    
### ECAL DQM MODULES ###

process.load("DQM.EcalCommon.EcalDQMBinningService_cfi")
'''
    if laser :
        ecalDQM += '''
process.load("DQM.EcalCalibration.EcalLaserMonitor_cfi")
'''
    else :
        ecalDQM += '''
process.load("DQM.EcalBarrelMonitorTasks.EcalMonitorTask_cfi")
process.load("DQM.EcalBarrelMonitorClient.EcalMonitorClient_cfi")
'''        

else :
    ecalDQM += '''
    
### ECAL DQM MODULES ###

process.load("DQM.EcalBarrelMonitorModule.EcalBarrelMonitorModule_cfi")
process.load("DQM.EcalEndcapMonitorModule.EcalEndcapMonitorModule_cfi")
process.load("DQM.EcalBarrelMonitorTasks.EcalBarrelMonitorTasks_cfi")
process.load("DQM.EcalEndcapMonitorTasks.EcalEndcapMonitorTasks_cfi")
'''

if not newFramework and not central and physics :
    ecalDQM += '''
process.load("DQM.EcalBarrelMonitorTasks.EBHltTask_cfi")
process.load("DQM.EcalEndcapMonitorTasks.EEHltTask_cfi")
'''

if not newFramework and physics and live :
    ecalDQM += '''
process.load("DQM.EcalBarrelMonitorTasks.EBTrendTask_cfi")
process.load("DQM.EcalBarrelMonitorClient.EBTrendClient_cfi")
process.load("DQM.EcalEndcapMonitorTasks.EETrendTask_cfi")
process.load("DQM.EcalEndcapMonitorClient.EETrendClient_cfi")
'''

if not newFramework :
    ecalDQM += '''
process.load("DQM.EcalBarrelMonitorClient.EcalBarrelMonitorClient_cfi")
process.load("DQM.EcalEndcapMonitorClient.EcalEndcapMonitorClient_cfi")
'''

if privEcal and not live :
    ecalDQM += '''
###### Temporary solution for merging the plots #####
process.load("Toolset.DQMTools.DQMFileLoader_cfi")
'''
    

dqmModules += '''

### DQM COMMON MODULES ###

process.load("DQMServices.Core.DQM_cfg")
'''

if not newFramework :
    dqmModules += '''
process.dqmQTestEB = cms.EDAnalyzer("QualityTester",
  reportThreshold = cms.untracked.string("red"),
  prescaleFactor = cms.untracked.int32(1),
  qtList = cms.untracked.FileInPath("DQM/EcalBarrelMonitorModule/test/data/EcalBarrelQualityTests.xml"),
  getQualityTestsFromFile = cms.untracked.bool(True),
  qtestOnEndLumi = cms.untracked.bool(True),
  qtestOnEndRun = cms.untracked.bool(True)
)
process.dqmQTestEE = cms.EDAnalyzer("QualityTester",
  reportThreshold = cms.untracked.string("red"),
  prescaleFactor = cms.untracked.int32(1),
  qtList = cms.untracked.FileInPath("DQM/EcalEndcapMonitorModule/test/data/EcalEndcapQualityTests.xml"),
  getQualityTestsFromFile = cms.untracked.bool(True),
  qtestOnEndLumi = cms.untracked.bool(True),
  qtestOnEndRun = cms.untracked.bool(True)
)
'''

if p5 :
    dqmModules += '''
process.load("DQM.Integration.test.environment_cfi")'''
else :
    dqmModules += '''
process.load("DQMServices.Components.DQMEnvironment_cfi")'''

if not newFramework :
    dqmModules += '''
process.dqmEnvEB = process.dqmEnv.clone()
process.dqmEnvEE = process.dqmEnv.clone()
'''


filters += '''

### FILTERS ###

process.load("FWCore.Modules.preScaler_cfi")
'''

if live :
    filters += '''
process.load("HLTrigger.special.HLTTriggerTypeFilter_cfi")
'''

if physics :
    filters += '''
process.ecalPhysicsFilter = cms.EDFilter("EcalMonitorPrescaler")
'''
elif calib :
    filters += '''
process.ecalCalibrationFilter = cms.EDFilter("EcalMonitorPrescaler")
process.ecalLaserFilter = cms.EDFilter("EcalMonitorPrescaler")
process.ecalLedFilter = cms.EDFilter("EcalMonitorPrescaler")
process.ecalPedestalFilter = cms.EDFilter("EcalMonitorPrescaler")
process.ecalTestPulseFilter = cms.EDFilter("EcalMonitorPrescaler")
'''


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
'''
    if live :
        sequencePaths += '#    process.hltTriggerTypeFilter +'

    sequencePaths += '''
    process.ecalEBunpacker
)

process.ecalRecoSequence = cms.Sequence(
    process.ecalUncalibHit *
    process.ecalDetIdToBeRecovered *
    process.ecalRecHit
)
'''

    if physics :
        sequencePaths += '''
process.ecalClusterSequence = cms.Sequence(
    process.hybridClusteringSequence +
    process.multi5x5ClusteringSequence
)
process.ecalClusterSequence.remove(process.multi5x5SuperClustersWithPreshower)
'''

    if not newFramework :
        sequencePaths += '''
process.ecalMonitorBaseSequence = cms.Sequence('''

#should run all files in one job
        if privEcal and not live :
            sequencePaths += '''
    process.dqmFileLoader +'''
        
        sequencePaths += '''
    process.ecalBarrelMonitorModule +
    process.ecalEndcapMonitorModule +
    process.ecalBarrelOccupancyTask +
    process.ecalBarrelIntegrityTask +
    process.ecalEndcapOccupancyTask +
    process.ecalEndcapIntegrityTask +
    process.ecalBarrelStatusFlagsTask +
    process.ecalBarrelRawDataTask +
    process.ecalEndcapStatusFlagsTask +
    process.ecalEndcapRawDataTask +
    process.ecalBarrelPedestalOnlineTask +
    process.ecalEndcapPedestalOnlineTask
)
'''

    if physics :
        sequencePaths += '''
process.ecalMonitorSequence = cms.Sequence('''

        if newFramework :
            sequencePaths += '''
    process.ecalMonitorTask'''
        else :
            if not central :
                sequencePaths += '''
    process.ecalBarrelHltTask +
    process.ecalEndcapHltTask +'''

            if live :
                sequencePaths += '''
    process.ecalBarrelTrendTask +
    process.ecalEndcapTrendTask +'''

            sequencePaths += '''
    process.ecalBarrelCosmicTask +
    process.ecalBarrelClusterTask +
    process.ecalBarrelTriggerTowerTask +
    process.ecalBarrelTimingTask +
    process.ecalBarrelSelectiveReadoutTask +
    process.ecalEndcapCosmicTask +
    process.ecalEndcapClusterTask +
    process.ecalEndcapTriggerTowerTask +
    process.ecalEndcapTimingTask +
    process.ecalEndcapSelectiveReadoutTask'''

        sequencePaths += '''
)

process.ecalMonitorPath = cms.Path(
    process.ecalPreRecoSequence *
    process.ecalPhysicsFilter *
    process.ecalRecoSequence *
    (
    process.ecalClusterSequence +'''

        if physics and live :
            sequencePaths += '''
    process.l1GtEvmUnpack +'''

        sequencePaths += '''    
    process.simEcalTriggerPrimitiveDigis
    ) *
    ('''
        if not newFramework :
            sequencePaths += '''
    process.ecalMonitorBaseSequence +'''

        sequencePaths += '''
    process.ecalMonitorSequence
    )
)
'''
    
    elif calib :
        sequencePaths += '''
process.ecalLaserPath = cms.Path(
    process.ecalPreRecoSequence *
    process.ecalLaserFilter *    
    process.ecalRecoSequence *
    process.ecalUncalibHit1 *
    (
    process.ecalMonitorBaseSequence +
    process.ecalBarrelLaserTask +
    process.ecalEndcapLaserTask
    )
)

process.ecalLedPath = cms.Path(
    process.ecalPreRecoSequence *
    process.ecalLedFilter *
    process.ecalRecoSequence *
    process.ecalUncalibHit1 *
    (
    process.ecalMonitorBaseSequence +    
    process.ecalEndcapLedTask
    )
)
'''
        if (daqtype == 'localDAQ') :
            sequencePaths += '''
process.ecalPedestalPath = cms.Path(
    process.ecalPreRecoSequence *
    process.ecalPedestalFilter *    
    process.ecalRecoSequence *
    (
    process.ecalMonitorBaseSequence +    
    process.ecalBarrelPedestalTask +
    process.ecalEndcapPedestalTask
    )
)    
'''

        sequencePaths += '''
process.ecalTestPulsePath = cms.Path(
    process.ecalPreRecoSequence *
    process.ecalTestPulseFilter *    
    process.ecalRecoSequence *
    process.ecalUncalibHit2 *
    (
    process.ecalMonitorBaseSequence +    
    process.ecalBarrelTestPulseTask +
    process.ecalEndcapTestPulseTask
    )
)
'''

    sequencePaths += '''
process.ecalClientPath = cms.Path(
    process.ecalPreRecoSequence *'''

    if physics :
        sequencePaths += '''
    process.ecalPhysicsFilter +'''
    else :
        sequencePaths += '''
    process.ecalCalibrationFilter +'''

    if not newFramework and physics and live :
        sequencePaths += '''
    process.ecalBarrelTrendClient +
    process.ecalEndcapTrendClient +'''

    if newFramework :
        sequencePaths += '''
    process.ecalMonitorClient'''
    else :
        sequencePaths += '''
    process.ecalBarrelMonitorClient +
    process.ecalEndcapMonitorClient'''

    sequencePaths += '''
)
'''
#end if laser else

sequencePaths += '''
process.ecalMonitorEndPath = cms.EndPath('''
if newFramework :
    sequencePaths += '''
    process.dqmEnv'''
else :
    sequencePaths += '''
    process.dqmEnvEB +
    process.dqmEnvEE +
    process.dqmQTestEB +
    process.dqmQTestEE'''

sequencePaths += '''
)'''

if doOutput :
    sequencePaths += '''
process.dqmEndPath = cms.EndPath(
    process.dqmSaver
)
'''

sequencePaths += '''
process.schedule = cms.Schedule('''

if physics :
    sequencePaths += '''
    process.ecalMonitorPath,
    process.ecalClientPath,'''
elif calib :
    sequencePaths += '''
    process.ecalLaserPath,
    process.ecalLedPath,'''

    if (daqtype == 'localDAQ') :
        sequencePaths += '''
    process.ecalPedestalPath,'''

    sequencePaths += '''
    process.ecalTestPulsePath,
    process.ecalClientPath,'''
elif laser :
    sequencePaths += '''
    process.ecalMonitorPath,'''

sequencePaths += '''
    process.ecalMonitorEndPath'''

if doOutput :
    sequencePaths += ''',
    process.dqmEndPath'''

sequencePaths += '''
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

process.ecalUncalibHit.EBdigiCollection = "ecalEBunpacker:ebDigis"
process.ecalUncalibHit.EEdigiCollection = "ecalEBunpacker:eeDigis"

process.ecalDetIdToBeRecovered.ebSrFlagCollection = "ecalEBunpacker"
process.ecalDetIdToBeRecovered.eeSrFlagCollection = "ecalEBunpacker"
process.ecalDetIdToBeRecovered.ebIntegrityGainErrors = "ecalEBunpacker:EcalIntegrityGainErrors"
process.ecalDetIdToBeRecovered.ebIntegrityGainSwitchErrors = "ecalEBunpacker:EcalIntegrityGainSwitchErrors"
process.ecalDetIdToBeRecovered.ebIntegrityChIdErrors = "ecalEBunpacker:EcalIntegrityChIdErrors"
process.ecalDetIdToBeRecovered.eeIntegrityGainErrors = "ecalEBunpacker:EcalIntegrityGainErrors"
process.ecalDetIdToBeRecovered.eeIntegrityGainSwitchErrors = "ecalEBunpacker:EcalIntegrityGainSwitchErrors"
process.ecalDetIdToBeRecovered.eeIntegrityChIdErrors = "ecalEBunpacker:EcalIntegrityChIdErrors"
process.ecalDetIdToBeRecovered.integrityTTIdErrors = "ecalEBunpacker:EcalIntegrityTTIdErrors"
process.ecalDetIdToBeRecovered.integrityBlockSizeErrors = "ecalEBunpacker:EcalIntegrityBlockSizeErrors"

process.ecalRecHit.killDeadChannels = True
process.ecalRecHit.ChannelStatusToBeExcluded = [ 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 78, 142 ]
process.ecalRecHit.EBuncalibRecHitCollection = "ecalUncalibHit:EcalUncalibRecHitsEB"
process.ecalRecHit.EEuncalibRecHitCollection = "ecalUncalibHit:EcalUncalibRecHitsEE"
'''

if physics :
    customizations += '''
process.simEcalTriggerPrimitiveDigis.Label = "ecalEBunpacker"
process.simEcalTriggerPrimitiveDigis.InstanceEB = "ebDigis"
process.simEcalTriggerPrimitiveDigis.InstanceEE = "eeDigis"
'''

if calib :
    customizations += '''
process.ecalUncalibHit1.MinAmplBarrel = 12.
process.ecalUncalibHit1.MinAmplEndcap = 16.
process.ecalUncalibHit1.EBdigiCollection = "ecalEBunpacker:ebDigis"
process.ecalUncalibHit1.EEdigiCollection = "ecalEBunpacker:eeDigis"

process.ecalUncalibHit2.EBdigiCollection = "ecalEBunpacker:ebDigis"
process.ecalUncalibHit2.EEdigiCollection = "ecalEBunpacker:eeDigis"
'''

customizations += '''
 ## Filters ##
'''

if physics :
    customizations += '''
process.ecalPhysicsFilter.EcalRawDataCollection = cms.InputTag("ecalEBunpacker")
process.ecalPhysicsFilter.clusterPrescaleFactor = cms.untracked.int32(1)
'''
    if live :
        customizations += '''
process.hltTriggerTypeFilter.SelectedTriggerType = 1 # 0=random, 1=physics, 2=calibration, 3=technical
'''
        
if calib :
    customizations += '''
process.ecalCalibrationFilter.EcalRawDataCollection = cms.InputTag("ecalEBunpacker")
process.ecalCalibrationFilter.laserPrescaleFactor = cms.untracked.int32(1)
process.ecalCalibrationFilter.ledPrescaleFactor = cms.untracked.int32(1)
process.ecalCalibrationFilter.pedestalPrescaleFactor = cms.untracked.int32(1)
process.ecalCalibrationFilter.testpulsePrescaleFactor = cms.untracked.int32(1)
    
process.ecalLaserFilter.EcalRawDataCollection = cms.InputTag("ecalEBunpacker")
process.ecalLaserFilter.laserPrescaleFactor = cms.untracked.int32(1)

process.ecalLedFilter.EcalRawDataCollection = cms.InputTag("ecalEBunpacker")
process.ecalLedFilter.ledPrescaleFactor = cms.untracked.int32(1)

process.ecalPedestalFilter.EcalRawDataCollection = cms.InputTag("ecalEBunpacker")
process.ecalPedestalFilter.pedestalPrescaleFactor = cms.untracked.int32(1)

process.ecalTestPulseFilter.EcalRawDataCollection = cms.InputTag("ecalEBunpacker")
process.ecalTestPulseFilter.testpulsePrescaleFactor = cms.untracked.int32(1)
'''
    if live :
        customizations += '''
process.hltTriggerTypeFilter.SelectedTriggerType = 2 # 0=random, 1=physics, 2=calibration, 3=technical
'''

customizations += '''
 ## Ecal DQM modules ##
'''

if newFramework :
    if physics :
        customizations += '''
process.ecalMonitorTask.taskParameters.Common.hltTaskMode = 2
process.ecalMonitorClient.runAtEndLumi = True
'''

    if laser :
        if privEcal :
            customizations += '''
process.ecalLaserMonitorClient.clientParameters.LightChecker.matacqPlotsDir = "/data/ecalod-disk01/dqm-data/laser"
'''

else :
    if calib :
        customizations += '''
process.ecalBarrelLaserTask.EcalUncalibratedRecHitCollection = "ecalUncalibHit1:EcalUncalibRecHitsEB"
process.ecalBarrelTestPulseTask.EcalUncalibratedRecHitCollection = "ecalUncalibHit2:EcalUncalibRecHitsEB"
process.ecalEndcapLaserTask.EcalUncalibratedRecHitCollection = "ecalUncalibHit1:EcalUncalibRecHitsEE"
process.ecalEndcapLedTask.EcalUncalibratedRecHitCollection = "ecalUncalibHit1:EcalUncalibRecHitsEE"
process.ecalEndcapTestPulseTask.EcalUncalibratedRecHitCollection = "ecalUncalibHit2:EcalUncalibRecHitsEE"

process.ecalBarrelLaserTask.laserWavelengths = [ 1, 2, 3, 4 ]
process.ecalEndcapLaserTask.laserWavelengths = [ 1, 2, 3, 4 ]
process.ecalEndcapLedTask.ledWavelengths = [ 1, 2 ]
process.ecalBarrelMonitorClient.laserWavelengths = [ 1, 2, 3, 4 ]
process.ecalEndcapMonitorClient.laserWavelengths = [ 1, 2, 3, 4 ]
process.ecalEndcapMonitorClient.ledWavelengths = [ 1, 2 ]

process.ecalBarrelMonitorClient.enabledClients = ["Integrity", "StatusFlags", "Occupancy", "PedestalOnline", "Pedestal", "TestPulse", "Laser", "Summary"]
process.ecalEndcapMonitorClient.enabledClients = ["Integrity", "StatusFlags", "Occupancy", "PedestalOnline","Pedestal", "TestPulse", "Laser", "Led", "Summary"]

process.ecalBarrelMonitorClient.produceReports = False
process.ecalEndcapMonitorClient.produceReports = False
'''

        if live :
            customizations += '''
process.ecalBarrelPedestalTask.MGPAGains = [ 12 ]
process.ecalBarrelPedestalTask.MGPAGainsPN = [ 16 ]
process.ecalBarrelTestPulseTask.MGPAGains = [ 12 ]
process.ecalBarrelTestPulseTask.MGPAGainsPN = [ 16 ]
process.ecalEndcapPedestalTask.MGPAGains = [ 12 ]
process.ecalEndcapPedestalTask.MGPAGainsPN = [ 16 ]
process.ecalEndcapTestPulseTask.MGPAGains = [ 12 ]
process.ecalEndcapTestPulseTask.MGPAGainsPN = [ 16 ]
process.ecalBarrelMonitorClient.MGPAGains = [ 12 ]
process.ecalBarrelMonitorClient.MGPAGainsPN = [ 16 ]
process.ecalEndcapMonitorClient.MGPAGains = [ 12 ]
process.ecalEndcapMonitorClient.MGPAGainsPN = [ 16 ]
'''

    if physics :
        customizations += '''
process.ecalBarrelMonitorClient.enabledClients = ["Integrity", "StatusFlags", "Occupancy", "PedestalOnline", "Timing", "Cosmic", "TriggerTower", "Cluster", "Summary"]
process.ecalEndcapMonitorClient.enabledClients = ["Integrity", "StatusFlags", "Occupancy", "PedestalOnline", "Timing", "Cosmic", "TriggerTower", "Cluster", "Summary"]

process.ecalBarrelMonitorClient.reducedReports = False
process.ecalEndcapMonitorClient.reducedReports = False
'''
        if live :
            customizations += '''
process.ecalBarrelTimingTask.useBeamStatus = cms.untracked.bool(True)
process.ecalEndcapTimingTask.useBeamStatus = cms.untracked.bool(True)
'''

    customizations += '''
process.ecalBarrelMonitorClient.location = "P5_Co"
process.ecalEndcapMonitorClient.location = "P5_Co"
process.ecalBarrelMonitorClient.verbose = True
process.ecalEndcapMonitorClient.verbose = True
'''

    if live :
        customizations += '''
process.ecalBarrelMonitorClient.updateTime = 4
process.ecalEndcapMonitorClient.updateTime = 4
'''

    if withDB :
        customizations += '''
os.environ["TNS_ADMIN"] = "/etc"
dbName = ""
dbHostName = ""
dbHostPort = 1521
dbUserName = ""
dbPassword = ""

try :
    file = open("/nfshome0/ecalpro/DQM/online-DQM/.cms_tstore.conf", "r")
    for line in file :
        if line.find("dbName") >= 0 :
            dbName = line.split()[2]
        if line.find("dbHostName") >= 0 :
            dbHostName = line.split()[2]
        if line.find("dbHostPort") >= 0 :
            dbHostPort = int(line.split()[2])
        if line.find("dbUserName") >= 0 :
            dbUserName = line.split()[2]
        if line.find("dbPassword") >= 0 :
            dbPassword = line.split()[2]
    file.close()
except IOError :
    pass

process.ecalBarrelMonitorClient.dbName = dbName
process.ecalBarrelMonitorClient.dbHostName = dbHostName
process.ecalBarrelMonitorClient.dbHostPort = dbHostPort
process.ecalBarrelMonitorClient.dbUserName = dbUserName
process.ecalBarrelMonitorClient.dbPassword = dbPassword

process.ecalEndcapMonitorClient.dbName = dbName
process.ecalEndcapMonitorClient.dbHostName = dbHostName
process.ecalEndcapMonitorClient.dbHostPort = dbHostPort
process.ecalEndcapMonitorClient.dbUserName = dbUserName
process.ecalEndcapMonitorClient.dbPassword = dbPassword
'''

    if (env == 'PrivLive') :
        customizations += '''
process.ecalBarrelMonitorClient.resetFile = "/data/ecalod-disk01/dqm-data/reset/EB"
process.ecalBarrelMonitorClient.resetFile = "/data/ecalod-disk01/dqm-data/reset/EE"
process.ecalBarrelMonitorClient.dbTagName = "CMSSW-online-private"
process.ecalEndcapMonitorClient.dbTagName = "CMSSW-online-private"
'''
    elif (env == 'PrivOffline') :
        customizations += '''
process.ecalBarrelMonitorClient.dbTagName = "CMSSW-offline-private"
process.ecalEndcapMonitorClient.dbTagName = "CMSSW-offline-private"
'''
    elif central :
        customizations += '''
process.ecalBarrelMonitorClient.dbUpdateTime = 120
process.ecalBarrelMonitorClient.dbUpdateTime = 120
process.ecalBarrelMonitorClient.dbTagName = "CMSSW-online-central"
process.ecalEndcapMonitorClient.dbTagName = "CMSSW-online-central"
'''

dirName = '/data/ecalod-disk01/dqm-data/tmp'

if privEcal and not live :
    customizations += 'process.dqmFileLoader.directory = "' + dirName + '"' + "\n"
    customizations += 'process.dqmFileLoader.workflow = "' + workflow + '"' + "\n"

customizations += '''
 ## DQM common modules ##
'''

if newFramework :
    if laser :
        customizations += '''
process.dqmEnv.subSystemFolder = cms.untracked.string("EcalCalibration")
'''
    else :
        customizations += '''
process.dqmEnv.subSystemFolder = cms.untracked.string("Ecal")
'''

else :
    if physics :
        customizations += '''
process.dqmEnvEB.subSystemFolder = cms.untracked.string("EcalBarrel")
process.dqmEnvEE.subSystemFolder = cms.untracked.string("EcalEndcap")
'''
    else :
        customizations += '''
process.dqmEnvEB.subSystemFolder = cms.untracked.string("EcalBarrel/Calibration")
process.dqmEnvEE.subSystemFolder = cms.untracked.string("EcalEndcap/Calibration")
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
        # temporary - remove when subsystemFolder issue is resolved
            if physics :
                customizations += 'process.dqmSaver.version = 1' + "\n"
            elif calib :
                customizations += 'process.dqmSaver.version = 2' + "\n"
            
        else :
            customizations += '''
process.dqmSaver.dirName = "''' + dirName + '''"
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
    customizations += 'process.source.consumerName = cms.untracked.string("' + workspace + ' DQM Consumer")' + "\n"
    if privEcal :
        customizations += 'process.source.sourceURL = cms.string("http://dqm-c2d07-30.cms:22100/urn:xdaq-application:lid=30")' + "\n"
    elif local :
        customizations += 'process.source.sourceURL = cms.string("http://localhost:22100/urn:xdaq-application:lid=30")' + "\n"

    if physics and (daqtype == 'globalDAQ') :
        customizations += 'process.source.SelectHLTOutput = cms.untracked.string("hltOutputA")' + "\n"
    else :
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
    customizations += '''
if process.runType.getRunType() == process.runType.cosmic_run :'''
    if newFramework :
        customizations += '''
    pass'''
    else :
        customizations += '''
    process.ecalMonitorEndPath.remove(process.dqmQTestEB)
    process.ecalMonitorEndPath.remove(process.dqmQTestEE)'''

    customizations += '''
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
process.ecalEBunpacker.InputLabel = cms.InputTag(FedRawData)
'''

    if not newFramework :
        customizations += '''
process.ecalBarrelRawDataTask.FEDRawDataCollection = cms.InputTag(FedRawData)
process.ecalEndcapRawDataTask.FEDRawDataCollection = cms.InputTag(FedRawData)
'''

if physics :
    if newFramework :
        customizations += 'process.ecalMonitorTask.collectionTags.Source = FedRawData' + "\n"
        if live :
            customizations += 'process.l1GtEvmUnpack.EvmGtInputTag = cms.InputTag(FedRawData)' + "\n"
            
    else :
        customizations += 'process.ecalBarrelSelectiveReadoutTask.FEDRawDataCollection = cms.InputTag(FedRawData)' + "\n"
        customizations += 'process.ecalEndcapSelectiveReadoutTask.FEDRawDataCollection = cms.InputTag(FedRawData)' + "\n"
        if not central :
            customizations += 'process.ecalBarrelHltTask.FEDRawDataCollection = cms.InputTag(FedRawData)' + "\n"
            customizations += 'process.ecalEndcapHltTask.FEDRawDataCollection = cms.InputTag(FedRawData)' + "\n"

        if live :
            customizations += 'process.l1GtEvmUnpack.EvmGtInputTag = cms.InputTag(FedRawData)' + "\n"
            customizations += 'process.ecalBarrelTrendTask.FEDRawDataCollection = cms.InputTag(FedRawData)' + "\n"
            customizations += 'process.ecalEndcapTrendTask.FEDRawDataCollection = cms.InputTag(FedRawData)' + "\n"

if not newFramework and calib and (daqtype == 'globalDAQ') :
    customizations += '''
 ## Avoid plot name clashes ##
process.ecalBarrelIntegrityTask.subfolder = "Calibration"
process.ecalBarrelOccupancyTask.subfolder = "Calibration"
process.ecalBarrelStatusFlagsTask.subfolder = "Calibration"
process.ecalBarrelRawDataTask.subfolder = "Calibration"
process.ecalBarrelPedestalOnlineTask.subfolder = "Calibration"
process.ecalEndcapIntegrityTask.subfolder = "Calibration"
process.ecalEndcapOccupancyTask.subfolder = "Calibration"
process.ecalEndcapStatusFlagsTask.subfolder = "Calibration"
process.ecalEndcapRawDataTask.subfolder = "Calibration"
process.ecalEndcapPedestalOnlineTask.subfolder = "Calibration"
process.ecalBarrelMonitorClient.subfolder = "Calibration"
process.ecalEndcapMonitorClient.subfolder = "Calibration"
'''

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

    if newFramework and not laser :
        e = 'new-' + e

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
