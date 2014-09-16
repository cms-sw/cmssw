import FWCore.ParameterSet.Config as cms

import os, sys, socket

liveECAL  = 0

localDAQ  = 0
globalDAQ = 0

ecalHostEC    = 'vmepcS2F19-25'.lower()

ecalHostEB    = 'srv-s2f19-26'.lower()
ecalHostEE    = 'srv-s2f19-27'.lower()

cmsLiveHostEB = 'dqm-c2d07-03'.lower()
cmsLiveHostEE = 'dqm-c2d07-04'.lower()

cmsPlayHostEB = 'dqm-c2d07-13'.lower()
cmsPlayHostEE = 'dqm-c2d07-14'.lower()

host = socket.gethostname().split('.')[0].lower()

print "Running on host:", host

if (host == ecalHostEC) :
  print "ERROR: Wrong host configuration."
  sys.exit(1)

liveCMS = 0
playCMS = 0

if (host == ecalHostEB) | (host == ecalHostEE) :
  if (liveECAL == 1) :
    if (localDAQ == 1) | (globalDAQ == 1) :
      print "ERROR: Wrong Live/Local/Global configuration."
      sys.exit(1)
  else :
    if (localDAQ == 1) & (globalDAQ == 1) :
      print "ERROR: Wrong Local/Global configuration."
      sys.exit(1)
  if (localDAQ == 1) :
    print "Running Ecal Local DQM."
  elif (globalDAQ == 1) :
    print "Running Ecal Global DQM."
  elif (liveECAL == 1) :
    print "Running Ecal Live DQM."
  else :
    print "ERROR: Unknown Ecal configuration."
    sys.exit(1)

if (host == cmsLiveHostEB) | (host == cmsLiveHostEE) :
  liveECAL  = 0
  localDAQ  = 0
  globalDAQ = 0
  liveCMS   = 1
  playCMS   = 0
  print "Running Central Live DQM."

if (host == cmsPlayHostEB) | (host == cmsPlayHostEE) :
  liveECAL  = 0
  localDAQ  = 0
  globalDAQ = 0
  liveCMS   = 0
  playCMS   = 1
  print "Running Central Playback DQM."

onlyEB = 0
onlyEE = 0

if (liveECAL == 1) | (liveCMS == 1) | (playCMS == 1) :
  if (host == ecalHostEB) | (host == cmsLiveHostEB) | (host == cmsPlayHostEB) :
    onlyEB = 1
    onlyEE = 0
    print "Running EB only."

if (liveECAL == 1) | (liveCMS == 1) | (playCMS == 1) :
  if (host == ecalHostEE) | (host == cmsLiveHostEE) | (host == cmsPlayHostEE) :
    onlyEB = 0
    onlyEE = 1
    print "Running EE only."

if (liveECAL == 1) | (liveCMS == 1) | (playCMS == 1) :
  if (onlyEB == 1) & (onlyEE == 1) :
    print "ERROR: Wrong EB/EE configuration."
    sys.exit(1)

if (localDAQ == 0) & (globalDAQ == 0) & (liveECAL == 0) & (liveCMS == 0) & (playCMS == 0) :
  print "ERROR: Unknown configuration."
  sys.exit(1)

os.environ['TNS_ADMIN'] = '/etc'

dbName = ''
dbHostName = ''
dbHostPort = 1521
dbUserName = ''
dbPassword = ''

if (localDAQ == 1) | (liveECAL == 1) | (liveCMS == 1) :
  try :
    file = open('/nfshome0/ecalpro/DQM/online-DQM/.cms_tstore.conf', 'r')
    for line in file :
      if line.find('dbName') >= 0 :
        dbName = line.split()[2]
      if line.find('dbHostName') >= 0 :
        dbHostName = line.split()[2]
      if line.find('dbHostPort') >= 0 :
        dbHostPort = int(line.split()[2])
      if line.find('dbUserName') >= 0 :
        dbUserName = line.split()[2]
      if line.find('dbPassword') >= 0 :
        dbPassword = line.split()[2]
    file.close()
  except IOError :
    pass

process = cms.Process("ECALDQM")

process.load("EventFilter.EcalRawToDigi.EcalUnpackerMapping_cfi")

process.load("EventFilter.EcalRawToDigi.EcalUnpackerData_cfi")

if (onlyEB == 1) :
  process.ecalEBunpacker.FEDs = [ 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645 ]
if (onlyEE == 1) :
  process.ecalEBunpacker.FEDs = [ 601, 602, 603, 604, 605, 606, 607, 608, 609, 646, 647, 648, 649, 650, 651, 652, 653, 654 ]

import RecoLocalCalo.EcalRecProducers.ecalMultiFitUncalibRecHit_cfi
process.ecalUncalibHit = RecoLocalCalo.EcalRecProducers.ecalMultiFitUncalibRecHit_cfi.ecalMultiFitUncalibRecHit.clone()

import RecoLocalCalo.EcalRecProducers.ecalFixedAlphaBetaFitUncalibRecHit_cfi
process.ecalUncalibHit1 = RecoLocalCalo.EcalRecProducers.ecalFixedAlphaBetaFitUncalibRecHit_cfi.ecalFixedAlphaBetaFitUncalibRecHit.clone()

import RecoLocalCalo.EcalRecProducers.ecalMaxSampleUncalibRecHit_cfi
process.ecalUncalibHit2 = RecoLocalCalo.EcalRecProducers.ecalMaxSampleUncalibRecHit_cfi.ecalMaxSampleUncalibRecHit.clone()

process.load("RecoLocalCalo.EcalRecProducers.ecalDetIdToBeRecovered_cfi")

process.load("RecoLocalCalo.EcalRecProducers.ecalRecHit_cfi")

process.load("Geometry.CaloEventSetup.CaloGeometry_cfi")

process.load("Geometry.CaloEventSetup.CaloTopology_cfi")

process.load("Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi")

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.load("Geometry.EcalMapping.EcalMapping_cfi")

process.load("Geometry.EcalMapping.EcalMappingRecord_cfi")

process.load("RecoEcal.EgammaClusterProducers.ecalClusteringSequence_cff")

process.load("CalibCalorimetry.EcalLaserCorrection.ecalLaserCorrectionService_cfi")

process.load("SimCalorimetry.EcalTrigPrimProducers.ecalTrigPrimESProducer_cff")
process.load("SimCalorimetry.EcalTrigPrimProducers.ecalTriggerPrimitiveDigis_readDBOffline_cff")

process.load("DQM.EcalBarrelMonitorModule.EcalBarrelMonitorModule_cfi")

process.load("DQM.EcalEndcapMonitorModule.EcalEndcapMonitorModule_cfi")

process.load("DQM.EcalBarrelMonitorTasks.EcalBarrelMonitorTasks_cfi")

process.load("DQM.EcalEndcapMonitorTasks.EcalEndcapMonitorTasks_cfi")

if (localDAQ == 1) :
  process.load("DQM.EcalBarrelMonitorClient.EcalBarrelMonitorClient_cfi")

  process.load("DQM.EcalEndcapMonitorClient.EcalEndcapMonitorClient_cfi")

if (globalDAQ == 1) | (liveECAL == 1) | (liveCMS == 1) | (playCMS == 1) :
  import DQM.EcalBarrelMonitorClient.EcalBarrelMonitorDbClient_cfi
  process.ecalBarrelMonitorClient = DQM.EcalBarrelMonitorClient.EcalBarrelMonitorDbClient_cfi.ecalBarrelMonitorDbClient.clone()

  import DQM.EcalEndcapMonitorClient.EcalEndcapMonitorDbClient_cfi
  process.ecalEndcapMonitorClient = DQM.EcalEndcapMonitorClient.EcalEndcapMonitorDbClient_cfi.ecalEndcapMonitorDbClient.clone()

process.load("DQMServices.Core.DQM_cfg")

if (localDAQ == 1) | (globalDAQ == 1) | (liveECAL == 1) :
  process.dqmSaver = cms.EDAnalyzer("DQMFileSaver",
    saveByRun = cms.untracked.int32(1),
    dirName = cms.untracked.string('.'),
    convention = cms.untracked.string('Online')
  )

  process.dqmEnv = cms.EDAnalyzer("DQMEventInfo",
    subSystemFolder = cms.untracked.string('Ecal')
  )

if (liveCMS == 1) | (playCMS == 1) :
  process.load("DQMServices.Components.DQMEnvironment_cfi")
  process.load("DQM.Integration.test.environment_cfi")

process.dqmInfoEB = cms.EDAnalyzer("DQMEventInfo",
  subSystemFolder = cms.untracked.string('EcalBarrel')
)

process.dqmInfoEE = cms.EDAnalyzer("DQMEventInfo",
  subSystemFolder = cms.untracked.string('EcalEndcap')
)

if (localDAQ == 1) :
  process.dqmSaver.referenceHandling = cms.untracked.string('skip')

if (globalDAQ == 1) | (liveECAL == 1) :
  process.dqmSaver.referenceHandling = cms.untracked.string('all')

if (liveCMS == 1) | (playCMS == 1) :
  process.dqmSaver.referenceHandling = 'all'

process.dqmQTestEB = cms.EDAnalyzer("QualityTester",
  reportThreshold = cms.untracked.string('red'),
  prescaleFactor = cms.untracked.int32(1),
  qtList = cms.untracked.FileInPath('DQM/EcalBarrelMonitorModule/test/data/EcalBarrelQualityTests.xml'),
  getQualityTestsFromFile = cms.untracked.bool(True),
  qtestOnEndLumi = cms.untracked.bool(False),
  qtestOnEndRun = cms.untracked.bool(True)
)

process.dqmQTestEE = cms.EDAnalyzer("QualityTester",
  reportThreshold = cms.untracked.string('red'),
  prescaleFactor = cms.untracked.int32(1),
  qtList = cms.untracked.FileInPath('DQM/EcalEndcapMonitorModule/test/data/EcalEndcapQualityTests.xml'),
  getQualityTestsFromFile = cms.untracked.bool(True),
  qtestOnEndLumi = cms.untracked.bool(False),
  qtestOnEndRun = cms.untracked.bool(True)
)

process.load("HLTrigger.special.HLTTriggerTypeFilter_cfi")

# 0=random, 1=physics, 2=calibration, 3=technical
process.hltTriggerTypeFilter.SelectedTriggerType = 1

process.load("FWCore.Modules.preScaler_cfi")

process.ecalPhysicsFilter = cms.EDFilter("EcalMonitorPrescaler",
  EcalRawDataCollection = cms.InputTag("ecalEBunpacker"),
  cosmicPrescaleFactor = cms.untracked.int32(1),
  physicsPrescaleFactor = cms.untracked.int32(1),
  laserPrescaleFactor = cms.untracked.int32(0),
  ledPrescaleFactor = cms.untracked.int32(0),
  pedestalPrescaleFactor = cms.untracked.int32(0),
  testpulsePrescaleFactor = cms.untracked.int32(0),
  pedestaloffsetPrescaleFactor = cms.untracked.int32(0)
)

process.ecalLaserLedFilter = cms.EDFilter("EcalMonitorPrescaler",
  EcalRawDataCollection = cms.InputTag("ecalEBunpacker"),
  cosmicPrescaleFactor = cms.untracked.int32(0),
  physicsPrescaleFactor = cms.untracked.int32(0),
  laserPrescaleFactor = cms.untracked.int32(1),
  ledPrescaleFactor = cms.untracked.int32(1),
  pedestalPrescaleFactor = cms.untracked.int32(0),
  testpulsePrescaleFactor = cms.untracked.int32(0),
  pedestaloffsetPrescaleFactor = cms.untracked.int32(0)
)

process.ecalPedestalFilter = cms.EDFilter("EcalMonitorPrescaler",
  EcalRawDataCollection = cms.InputTag("ecalEBunpacker"),
  cosmicPrescaleFactor = cms.untracked.int32(0),
  physicsPrescaleFactor = cms.untracked.int32(0),
  laserPrescaleFactor = cms.untracked.int32(0),
  ledPrescaleFactor = cms.untracked.int32(0),
  pedestalPrescaleFactor = cms.untracked.int32(1),
  testpulsePrescaleFactor = cms.untracked.int32(0),
  pedestaloffsetPrescaleFactor = cms.untracked.int32(1)
)

process.ecalTestPulseFilter = cms.EDFilter("EcalMonitorPrescaler",
  EcalRawDataCollection = cms.InputTag("ecalEBunpacker"),
  cosmicPrescaleFactor = cms.untracked.int32(0),
  physicsPrescaleFactor = cms.untracked.int32(0),
  laserPrescaleFactor = cms.untracked.int32(0),
  ledPrescaleFactor = cms.untracked.int32(0),
  pedestalPrescaleFactor = cms.untracked.int32(0),
  testpulsePrescaleFactor = cms.untracked.int32(1),
  pedestaloffsetPrescaleFactor = cms.untracked.int32(0)
)

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(-1)
)

if (localDAQ == 1) | (globalDAQ == 1) :
  process.source = cms.Source("NewEventStreamFileReader",
    fileNames = cms.untracked.vstring('file:file.dat')
  )

if (liveECAL == 1) :
  process.source = cms.Source("EventStreamHttpReader",
    sourceURL = cms.string('http://dqm-c2d07-30.cms:22100/urn:xdaq-application:lid=30'),
    consumerPriority = cms.untracked.string('normal'),
    max_event_size = cms.int32(7000000),
    consumerName = cms.untracked.string('Ecal DQM Consumer'),
    SelectHLTOutput = cms.untracked.string('hltOutputDQM'),
    max_queue_depth = cms.int32(5),
    maxEventRequestRate = cms.untracked.double(100.0),
    SelectEvents = cms.untracked.PSet( SelectEvents = cms.vstring('*') ),
    headerRetryInterval = cms.untracked.int32(3)
  )

if (liveCMS == 1) :
  process.load("DQM.Integration.test.inputsource_cfi")

if (playCMS == 1) :
  process.load("DQM.Integration.test.inputsource_cfi")

if (localDAQ == 1) | (globalDAQ == 1) | (liveECAL == 1) :
  process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
  process.GlobalTag.connect = "frontier://(proxyurl=http://localhost:3128)(serverurl=http://frontier1.cms:8000/FrontierOnProd)(serverurl=http://frontier2.cms:8000/FrontierOnProd)(retrieve-ziplevel=0)/CMS_COND_31X_GLOBALTAG"
  process.GlobalTag.globaltag = "GR10_H_V6A::All"
  process.prefer("GlobalTag")

if (liveCMS == 1) | (playCMS == 1) :
  process.load("DQM.Integration.test.FrontierCondition_GT_cfi")
  process.prefer("GlobalTag")

if (globalDAQ == 1) | (liveECAL == 1) | (liveCMS == 1) | (playCMS == 1) :
  process.GlobalTag.toGet = cms.VPSet(
    cms.PSet(record = cms.string("EcalDQMChannelStatusRcd"),
             tag = cms.string("EcalDQMChannelStatus_v1_hlt"),
             connect = cms.untracked.string("frontier://(proxyurl=http://localhost:3128)(serverurl=http://frontier1.cms:8000/FrontierOnProd)(serverurl=http://frontier2.cms:8000/FrontierOnProd)(retrieve-ziplevel=0)/CMS_COND_34X_ECAL")
            ),
    cms.PSet(record = cms.string("EcalDQMTowerStatusRcd"),
             tag = cms.string("EcalDQMTowerStatus_v1_hlt"),
             connect = cms.untracked.string("frontier://(proxyurl=http://localhost:3128)(serverurl=http://frontier1.cms:8000/FrontierOnProd)(serverurl=http://frontier2.cms:8000/FrontierOnProd)(retrieve-ziplevel=0)/CMS_COND_34X_ECAL")
            )
  )

process.MessageLogger = cms.Service("MessageLogger",
  cout = cms.untracked.PSet(
    threshold = cms.untracked.string('WARNING'),
    noLineBreaks = cms.untracked.bool(True),
    noTimeStamps = cms.untracked.bool(True),
    default = cms.untracked.PSet(
      limit = cms.untracked.int32(0)
    ),
    EcalRawToDigi = cms.untracked.PSet(
      limit = cms.untracked.int32(1000)
    ),
    EcalRawToDigiTriggerType = cms.untracked.PSet(
      limit = cms.untracked.int32(1000)
    ),
    EcalRawToDigiTpg = cms.untracked.PSet(
      limit = cms.untracked.int32(1000)
    ),
    EcalRawToDigiNumTowerBlocks = cms.untracked.PSet(
      limit = cms.untracked.int32(1000)
    ),
    EcalRawToDigiTowerId = cms.untracked.PSet(
      limit = cms.untracked.int32(1000)
    ),
    EcalRawToDigiTowerSize = cms.untracked.PSet(
      limit = cms.untracked.int32(1000)
    ),
    EcalRawToDigiChId = cms.untracked.PSet(
      limit = cms.untracked.int32(1000)
    ),
    EcalRawToDigiGainZero = cms.untracked.PSet(
      limit = cms.untracked.int32(1000)
    ),
    EcalRawToDigiGainSwitch = cms.untracked.PSet(
      limit = cms.untracked.int32(1000)
    ),
    EcalRawToDigiDccBlockSize = cms.untracked.PSet(
      limit = cms.untracked.int32(1000)
    ),
    EcalRawToDigiMemBlock = cms.untracked.PSet(
      limit = cms.untracked.int32(1000)
    ),
    EcalRawToDigiMemTowerId = cms.untracked.PSet(
      limit = cms.untracked.int32(1000)
    ),
    EcalRawToDigiMemChId = cms.untracked.PSet(
      limit = cms.untracked.int32(1000)
    ),
    EcalRawToDigiMemGain = cms.untracked.PSet(
      limit = cms.untracked.int32(1000)
    ),
    EcalRawToDigiTCC = cms.untracked.PSet(
      limit = cms.untracked.int32(1000)
    ),
    EcalRawToDigiSRP = cms.untracked.PSet(
      limit = cms.untracked.int32(1000)
    ),
    EcalDCCHeaderRuntypeDecoder = cms.untracked.PSet(
      limit = cms.untracked.int32(1000)
    ),
    EcalBarrelMonitorModule = cms.untracked.PSet(
      limit = cms.untracked.int32(1000)
    ),
    EcalEndcapMonitorModule = cms.untracked.PSet(
      limit = cms.untracked.int32(1000)
    )
  ),
  categories = cms.untracked.vstring('EcalRawToDigi',
                                     'EcalRawToDigiTriggerType',
                                     'EcalRawToDigiTpg',
                                     'EcalRawToDigiNumTowerBlocks',
                                     'EcalRawToDigiTowerId',
                                     'EcalRawToDigiTowerSize',
                                     'EcalRawToDigiChId',
                                     'EcalRawToDigiGainZero',
                                     'EcalRawToDigiGainSwitch',
                                     'EcalRawToDigiDccBlockSize',
                                     'EcalRawToDigiMemBlock',
                                     'EcalRawToDigiMemTowerId',
                                     'EcalRawToDigiMemChId',
                                     'EcalRawToDigiMemGain',
                                     'EcalRawToDigiTCC',
                                     'EcalRawToDigiSRP',
                                     'EcalDCCHeaderRuntypeDecoder',
                                     'EcalBarrelMonitorModule',
                                     'EcalEndcapMonitorModule'),
  destinations = cms.untracked.vstring('cout')
)

process.ModuleWebRegistry = cms.Service("ModuleWebRegistry")

process.preScaler.prescaleFactor = 1

if (liveECAL == 1) | (liveCMS == 1) | (playCMS == 1) :
  process.dqmQTestEB.prescaleFactor = 16
  process.dqmQTestEB.qtestOnEndLumi = True
  process.dqmQTestEE.prescaleFactor = 16
  process.dqmQTestEE.qtestOnEndLumi = True

process.ecalDataSequence = cms.Sequence(process.preScaler*process.ecalEBunpacker*process.ecalUncalibHit*process.ecalDetIdToBeRecovered*process.ecalRecHit)

process.ecalBarrelMonitorSequence = cms.Sequence(process.ecalBarrelMonitorModule*process.ecalBarrelMonitorClient)

process.ecalEndcapMonitorSequence = cms.Sequence(process.ecalEndcapMonitorModule*process.ecalEndcapMonitorClient)

process.load("DQM.EcalBarrelMonitorTasks.EBHltTask_cfi")
process.load("DQM.EcalBarrelMonitorTasks.EBTrendTask_cfi")
process.load("DQM.EcalBarrelMonitorClient.EBTrendClient_cfi")

process.ecalBarrelMainSequence = cms.Sequence(process.ecalBarrelPedestalOnlineTask*process.ecalBarrelOccupancyTask*process.ecalBarrelIntegrityTask*process.ecalBarrelStatusFlagsTask*process.ecalBarrelRawDataTask*process.ecalBarrelHltTask*process.ecalBarrelTrendTask*process.ecalBarrelTrendClient)

process.load("DQM.EcalEndcapMonitorTasks.EEHltTask_cfi")
process.load("DQM.EcalEndcapMonitorTasks.EETrendTask_cfi")
process.load("DQM.EcalEndcapMonitorClient.EETrendClient_cfi")

process.ecalEndcapMainSequence = cms.Sequence(process.ecalEndcapPedestalOnlineTask*process.ecalEndcapOccupancyTask*process.ecalEndcapIntegrityTask*process.ecalEndcapStatusFlagsTask*process.ecalEndcapRawDataTask*process.ecalEndcapHltTask*process.ecalEndcapTrendTask*process.ecalEndcapTrendClient)

process.ecalBarrelPhysicsSequence = cms.Sequence(process.ecalBarrelPedestalOnlineTask*process.ecalBarrelCosmicTask*process.ecalBarrelClusterTask*process.ecalBarrelTriggerTowerTask*process.ecalBarrelTimingTask*process.ecalBarrelSelectiveReadoutTask)

process.ecalEndcapPhysicsSequence = cms.Sequence(process.ecalEndcapPedestalOnlineTask*process.ecalEndcapCosmicTask*process.ecalEndcapClusterTask*process.ecalEndcapTriggerTowerTask*process.ecalEndcapTimingTask*process.ecalEndcapSelectiveReadoutTask)

if (liveCMS == 1) | (playCMS == 1) :
  process.ecalBarrelMainSequence.remove(process.ecalBarrelHltTask)

  process.ecalEndcapMainSequence.remove(process.ecalEndcapHltTask)

if (localDAQ == 1) :
  process.ecalBarrelPhysicsSequence.remove(process.ecalBarrelPedestalOnlineTask)

  process.ecalEndcapPhysicsSequence.remove(process.ecalEndcapPedestalOnlineTask)

if (globalDAQ == 1) | (liveECAL == 1) | (liveCMS == 1) | (playCMS == 1) :
  process.ecalBarrelMainSequence.remove(process.ecalBarrelPedestalOnlineTask)

  process.ecalEndcapMainSequence.remove(process.ecalEndcapPedestalOnlineTask)

process.ecalClusterSequence = cms.Sequence(process.hybridSuperClusters*process.correctedHybridSuperClusters*process.multi5x5BasicClusters*process.multi5x5SuperClusters)

process.ecalMonitorPath = cms.Path(process.ecalDataSequence*process.ecalBarrelMonitorSequence*process.ecalEndcapMonitorSequence)

process.ecalPhysicsPath = cms.Path(process.ecalDataSequence*process.ecalPhysicsFilter*process.hltTriggerTypeFilter*process.simEcalTriggerPrimitiveDigis*process.ecalClusterSequence*process.ecalBarrelMainSequence*process.ecalBarrelPhysicsSequence*process.ecalEndcapMainSequence*process.ecalEndcapPhysicsSequence)

if (localDAQ == 1) | (globalDAQ == 1) :
  process.ecalPhysicsPath.remove(process.hltTriggerTypeFilter)

if (liveECAL == 1) | (liveCMS == 1) | (playCMS == 1) :
  process.ecalPhysicsPath.remove(process.ecalPhysicsFilter)

process.ecalLaserLedPath = cms.Path(process.ecalDataSequence*process.ecalLaserLedFilter*process.ecalUncalibHit1*process.ecalBarrelMainSequence*process.ecalBarrelLaserTask*process.ecalEndcapMainSequence*process.ecalEndcapLaserTask*process.ecalEndcapLedTask)

process.ecalPedestalPath = cms.Path(process.ecalDataSequence*process.ecalPedestalFilter*process.ecalBarrelMainSequence*process.ecalBarrelPedestalTask*process.ecalEndcapMainSequence*process.ecalEndcapPedestalTask)

process.ecalTestPulsePath = cms.Path(process.ecalDataSequence*process.ecalTestPulseFilter*process.ecalUncalibHit2*process.ecalBarrelMainSequence*process.ecalBarrelTestPulseTask*process.ecalEndcapMainSequence*process.ecalEndcapTestPulseTask)

process.ecalMonitorEndPath = cms.EndPath(process.dqmEnv*process.dqmInfoEB*process.dqmInfoEE*process.dqmQTestEB*process.dqmQTestEE*process.dqmSaver)

process.schedule = cms.Schedule(process.ecalMonitorPath,process.ecalPhysicsPath,process.ecalLaserLedPath,process.ecalPedestalPath,process.ecalTestPulsePath,process.ecalMonitorEndPath)

if (liveECAL == 1) :
  process.ecalMonitorEndPath.remove(process.dqmEnv)

if (liveCMS == 1) | (playCMS == 1) :
  process.ecalMonitorEndPath.remove(process.dqmInfoEB)
  process.ecalMonitorEndPath.remove(process.dqmInfoEE)

if (onlyEE == 1) :
  process.ecalMonitorPath.remove(process.ecalBarrelMonitorSequence)
  process.ecalPhysicsPath.remove(process.ecalBarrelMainSequence)
  process.ecalPhysicsPath.remove(process.ecalBarrelPhysicsSequence)
  process.ecalLaserLedPath.remove(process.ecalBarrelMainSequence)
  process.ecalLaserLedPath.remove(process.ecalBarrelLaserTask)
  process.ecalPedestalPath.remove(process.ecalBarrelMainSequence)
  process.ecalPedestalPath.remove(process.ecalBarrelPedestalTask)
  process.ecalTestPulsePath.remove(process.ecalBarrelMainSequence)
  process.ecalTestPulsePath.remove(process.ecalBarrelTestPulseTask)
  process.ecalMonitorEndPath.remove(process.dqmInfoEB)
  process.ecalMonitorEndPath.remove(process.dqmQTestEB)

if (onlyEB == 1) :
  process.ecalMonitorPath.remove(process.ecalEndcapMonitorSequence)
  process.ecalPhysicsPath.remove(process.ecalEndcapMainSequence)
  process.ecalPhysicsPath.remove(process.ecalEndcapPhysicsSequence)
  process.ecalLaserLedPath.remove(process.ecalEndcapMainSequence)
  process.ecalLaserLedPath.remove(process.ecalEndcapLaserTask)
  process.ecalLaserLedPath.remove(process.ecalEndcapLedTask)
  process.ecalPedestalPath.remove(process.ecalEndcapMainSequence)
  process.ecalPedestalPath.remove(process.ecalEndcapPedestalTask)
  process.ecalTestPulsePath.remove(process.ecalEndcapMainSequence)
  process.ecalTestPulsePath.remove(process.ecalEndcapTestPulseTask)
  process.ecalMonitorEndPath.remove(process.dqmInfoEE)
  process.ecalMonitorEndPath.remove(process.dqmQTestEE)

if (liveECAL == 1) | (liveCMS == 1) | (playCMS == 1) :
  process.EventStreamHttpReader.SelectEvents = cms.untracked.PSet( SelectEvents = cms.vstring('*'))
#  process.EventStreamHttpReader.SelectEvents = cms.untracked.PSet( SelectEvents = cms.vstring('HLT_Physics','HLT_EcalCalibration'))
#  process.EventStreamHttpReader.SelectEvents = cms.untracked.PSet( SelectEvents = cms.vstring('HLT_MinBiasEcal','HLT_L1MuOpen','HLT_EcalCalibration'))

if (onlyEB == 1) :
  process.dqmEnv.subSystemFolder = 'EcalBarrel'
  if (liveECAL == 1) | (liveCMS == 1) | (playCMS == 1) :
    process.EventStreamHttpReader.consumerName = 'EcalBarrel DQM Consumer'

if (onlyEE == 1) :
  process.dqmEnv.subSystemFolder = 'EcalEndcap'
  if (liveECAL == 1) | (liveCMS == 1) | (playCMS == 1) :
    process.EventStreamHttpReader.consumerName = 'EcalEndcap DQM Consumer'

if (localDAQ == 1) | (globalDAQ == 1) :
  process.ecalEBunpacker.silentMode = False

process.ecalUncalibHit.EBdigiCollection = 'ecalEBunpacker:ebDigis'
process.ecalUncalibHit.EEdigiCollection = 'ecalEBunpacker:eeDigis'

process.ecalUncalibHit1.MinAmplBarrel = 12.
process.ecalUncalibHit1.MinAmplEndcap = 16.
process.ecalUncalibHit1.EBdigiCollection = 'ecalEBunpacker:ebDigis'
process.ecalUncalibHit1.EEdigiCollection = 'ecalEBunpacker:eeDigis'

process.ecalUncalibHit2.EBdigiCollection = 'ecalEBunpacker:ebDigis'
process.ecalUncalibHit2.EEdigiCollection = 'ecalEBunpacker:eeDigis'

process.ecalDetIdToBeRecovered.ebSrFlagCollection = 'ecalEBunpacker'
process.ecalDetIdToBeRecovered.eeSrFlagCollection = 'ecalEBunpacker'
process.ecalDetIdToBeRecovered.ebIntegrityGainErrors = 'ecalEBunpacker:EcalIntegrityGainErrors'
process.ecalDetIdToBeRecovered.ebIntegrityGainSwitchErrors = 'ecalEBunpacker:EcalIntegrityGainSwitchErrors'
process.ecalDetIdToBeRecovered.ebIntegrityChIdErrors = 'ecalEBunpacker:EcalIntegrityChIdErrors'
process.ecalDetIdToBeRecovered.eeIntegrityGainErrors = 'ecalEBunpacker:EcalIntegrityGainErrors'
process.ecalDetIdToBeRecovered.eeIntegrityGainSwitchErrors = 'ecalEBunpacker:EcalIntegrityGainSwitchErrors'
process.ecalDetIdToBeRecovered.eeIntegrityChIdErrors = 'ecalEBunpacker:EcalIntegrityChIdErrors'
process.ecalDetIdToBeRecovered.integrityTTIdErrors = 'ecalEBunpacker:EcalIntegrityTTIdErrors'
process.ecalDetIdToBeRecovered.integrityBlockSizeErrors = 'ecalEBunpacker:EcalIntegrityBlockSizeErrors'

process.ecalRecHit.killDeadChannels = True
process.ecalRecHit.ChannelStatusToBeExcluded = [ 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 78, 142 ]
process.ecalRecHit.EBuncalibRecHitCollection = 'ecalUncalibHit:EcalUncalibRecHitsEB'
process.ecalRecHit.EEuncalibRecHitCollection = 'ecalUncalibHit:EcalUncalibRecHitsEE'

process.ecalBarrelCosmicTask.EcalUncalibratedRecHitCollection = 'ecalUncalibHit:EcalUncalibRecHitsEB'

process.ecalBarrelLaserTask.EcalUncalibratedRecHitCollection = 'ecalUncalibHit1:EcalUncalibRecHitsEB'

process.ecalBarrelTestPulseTask.EcalUncalibratedRecHitCollection = 'ecalUncalibHit2:EcalUncalibRecHitsEB'

process.ecalBarrelTimingTask.EcalRecHitCollection = 'ecalRecHit:EcalRecHitsEB'

process.ecalEndcapCosmicTask.EcalUncalibratedRecHitCollection = 'ecalUncalibHit:EcalUncalibRecHitsEE'

process.ecalEndcapLaserTask.EcalUncalibratedRecHitCollection = 'ecalUncalibHit1:EcalUncalibRecHitsEE'

process.ecalEndcapLedTask.EcalUncalibratedRecHitCollection = 'ecalUncalibHit1:EcalUncalibRecHitsEE'

process.ecalEndcapTestPulseTask.EcalUncalibratedRecHitCollection = 'ecalUncalibHit2:EcalUncalibRecHitsEE'

process.ecalEndcapTimingTask.EcalRecHitCollection = 'ecalRecHit:EcalRecHitsEE'

process.simEcalTriggerPrimitiveDigis.Label = 'ecalEBunpacker'
process.simEcalTriggerPrimitiveDigis.InstanceEB = 'ebDigis'
process.simEcalTriggerPrimitiveDigis.InstanceEE = 'eeDigis'

if (localDAQ == 1) :
  process.ecalBarrelMonitorClient.maskFile = 'DQM/EcalBarrelMonitorModule/test/data/maskfile-EB.dat'

process.ecalBarrelMonitorClient.location = 'P5_Co'
process.ecalBarrelMonitorClient.enabledClients = ['Integrity', 'StatusFlags', 'Occupancy', 'PedestalOnline', 'Pedestal', 'TestPulse', 'Laser', 'Timing', 'Cosmic', 'TriggerTower', 'Cluster', 'Summary']

if (localDAQ == 1) :
  process.ecalEndcapMonitorClient.maskFile = 'DQM/EcalEndcapMonitorModule/test/data/maskfile-EE.dat'

process.ecalEndcapMonitorClient.location = 'P5_Co'
process.ecalEndcapMonitorClient.enabledClients = ['Integrity', 'StatusFlags', 'Occupancy', 'PedestalOnline', 'Pedestal', 'TestPulse', 'Laser', 'Led', 'Timing', 'Cosmic', 'TriggerTower', 'Cluster', 'Summary']

#process.ecalBarrelLaserTask.laserWavelengths = [ 1, 2, 3, 4 ]
process.ecalBarrelLaserTask.laserWavelengths = [ 1, 4 ]
#process.ecalBarrelLaserTask.laserWavelengths = [ 1 ]

#process.ecalEndcapLaserTask.laserWavelengths = [ 1, 2, 3, 4 ]
process.ecalEndcapLaserTask.laserWavelengths = [ 1, 4 ]
#process.ecalEndcapLaserTask.laserWavelengths = [ 1 ]

process.ecalEndcapLedTask.ledWavelengths = [ 1, 2 ]
#process.ecalEndcapLedTask.ledWavelengths = [ 1 ]

#process.ecalBarrelMonitorClient.laserWavelengths = [ 1, 2, 3, 4 ]
process.ecalBarrelMonitorClient.laserWavelengths = [ 1, 4 ]
#process.ecalBarrelMonitorClient.laserWavelengths = [ 1 ]

#process.ecalEndcapMonitorClient.laserWavelengths = [ 1, 2, 3, 4 ]
process.ecalEndcapMonitorClient.laserWavelengths = [ 1, 4 ]
#process.ecalEndcapMonitorClient.laserWavelengths = [ 1 ]

#process.ecalEndcapMonitorClient.ledWavelengths = [ 1 ]
process.ecalEndcapMonitorClient.ledWavelengths = [ 1, 2 ]

if (liveECAL == 1) | (liveCMS == 1) :
  #process.ecalBarrelPedestalTask.MGPAGains = [ 1, 6, 12 ]
  process.ecalBarrelPedestalTask.MGPAGains = [ 12 ]
  #process.ecalBarrelPedestalTask.MGPAGainsPN = [ 1, 16 ]
  process.ecalBarrelPedestalTask.MGPAGainsPN = [ 16 ]
  #process.ecalBarrelTestPulseTask.MGPAGains = [ 1, 6, 12 ]
  process.ecalBarrelTestPulseTask.MGPAGains = [ 12 ]
  #process.ecalBarrelTestPulseTask.MGPAGainsPN = [ 1, 16 ]
  process.ecalBarrelTestPulseTask.MGPAGainsPN = [ 16 ]

  #process.ecalEndcapPedestalTask.MGPAGains = [ 1, 6, 12 ]
  process.ecalEndcapPedestalTask.MGPAGains = [ 12 ]
  #process.ecalEndcapPedestalTask.MGPAGainsPN = [ 1, 16 ]
  process.ecalEndcapPedestalTask.MGPAGainsPN = [ 16 ]
  #process.ecalEndcapTestPulseTask.MGPAGains = [ 1, 6, 12 ]
  process.ecalEndcapTestPulseTask.MGPAGains = [ 12 ]
  #process.ecalEndcapTestPulseTask.MGPAGainsPN = [ 1, 16 ]
  process.ecalEndcapTestPulseTask.MGPAGainsPN = [ 16 ]

  #process.ecalBarrelMonitorClient.MGPAGains = [ 1, 6, 12 ]
  process.ecalBarrelMonitorClient.MGPAGains = [ 12 ]
  #process.ecalBarrelMonitorClient.MGPAGainsPN = [ 1, 16 ]
  process.ecalBarrelMonitorClient.MGPAGainsPN = [ 16 ]

  #process.ecalEndcapMonitorClient.MGPAGains = [ 1, 6, 12 ]
  process.ecalEndcapMonitorClient.MGPAGains = [ 12 ]
  #process.ecalEndcapMonitorClient.MGPAGainsPN = [ 1, 16 ]
  process.ecalEndcapMonitorClient.MGPAGainsPN = [ 16 ]

if (liveECAL == 1) :
  process.ecalBarrelMonitorClient.dbTagName = 'CMSSW-online-private'

  process.ecalEndcapMonitorClient.dbTagName = 'CMSSW-online-private'

if (liveCMS == 1) :
  process.ecalBarrelMonitorClient.dbTagName = 'CMSSW-online-central'

  process.ecalEndcapMonitorClient.dbTagName = 'CMSSW-online-central'

if (playCMS == 1) :
  process.ecalBarrelMonitorClient.dbTagName = 'CMSSW-offline-central'

  process.ecalEndcapMonitorClient.dbTagName = 'CMSSW-offline-central'

if (localDAQ == 1) | (globalDAQ == 1) :
  process.ecalBarrelMonitorClient.dbTagName = 'CMSSW-offline-private'

  process.ecalEndcapMonitorClient.dbTagName = 'CMSSW-offline-private'

if (localDAQ == 1) | (globalDAQ == 1) | (liveECAL == 1) | (liveCMS == 1) :
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

if (liveECAL == 1) :
  process.ecalBarrelMonitorClient.updateTime = 4
  process.ecalBarrelMonitorClient.dbUpdateTime = 120

  process.ecalEndcapMonitorClient.updateTime = 4
  process.ecalEndcapMonitorClient.dbUpdateTime = 120

if (liveCMS == 1) :
  process.ecalBarrelMonitorClient.updateTime = 4
  process.ecalBarrelMonitorClient.dbUpdateTime = 240
 
  process.ecalEndcapMonitorClient.updateTime = 4
  process.ecalEndcapMonitorClient.dbUpdateTime = 240

if (globalDAQ == 1) | (liveECAL == 1) :
  process.DQMStore.referenceFileName = '/data/ecalod-disk01/dqm-data/reference/REFERENCE.root'

if (liveCMS == 1) | (playCMS == 1) :
  if (onlyEB == 1) :
    process.DQMStore.referenceFileName = '/dqmdata/dqm/reference/eb_reference.root'
  if (onlyEE == 1) :
    process.DQMStore.referenceFileName = '/dqmdata/dqm/reference/ee_reference.root'

if (localDAQ == 1) | (globalDAQ == 1) :
  process.DQM.collectorHost = ''
  process.dqmSaver.dirName = '/data/ecalod-disk01/dqm-data/tmp'

if (liveECAL == 1) :
  process.DQM.collectorHost = 'ecalod-web01.cms'
  process.dqmSaver.dirName = '/data/ecalod-disk01/dqm-data/storage-manager/root'

