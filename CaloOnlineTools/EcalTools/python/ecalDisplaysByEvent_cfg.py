import FWCore.ParameterSet.Config as cms

process = cms.Process("ECALDISPLAYSBYEVENT")
process.load("EventFilter.EcalRawToDigi.EcalUnpackerMapping_cfi")
process.load("EventFilter.EcalRawToDigi.EcalUnpackerData_cfi")

process.load("Geometry.CaloEventSetup.CaloTopology_cfi")

process.load("Geometry.EcalCommonData.EcalOnly_cfi")

process.load("Geometry.CaloEventSetup.CaloGeometry_cff")
process.load("Geometry.EcalMapping.EcalMapping_cfi")
process.load("Geometry.EcalMapping.EcalMappingRecord_cfi")

process.load("CalibCalorimetry.EcalLaserCorrection.ecalLaserCorrectionService_cfi")

import RecoLocalCalo.EcalRecProducers.ecalMaxSampleUncalibRecHit_cfi
process.ecalUncalibHit = RecoLocalCalo.EcalRecProducers.ecalMaxSampleUncalibRecHit_cfi.ecalMaxSampleUncalibRecHit.clone()
process.ecalUncalibHit.EBdigiCollection = 'ecalEBunpacker:ebDigis'
process.ecalUncalibHit.EEdigiCollection = 'ecalEBunpacker:eeDigis'

process.load("RecoLocalCalo.EcalRecProducers.ecalRecHit_cfi")
process.ecalRecHit.ChannelStatusToBeExcluded = [1]
process.ecalRecHit.EBuncalibRecHitCollection = 'ecalUncalibHit:EcalUncalibRecHitsEB'
process.ecalRecHit.EEuncalibRecHitCollection = 'ecalUncalibHit:EcalUncalibRecHitsEE'

process.load("CaloOnlineTools.EcalTools.ecalDisplaysByEvent_cfi")
process.load("HLTrigger.special.TriggerTypeFilter_cfi")
process.triggerTypeFilter.SelectedTriggerType = 1

process.source = cms.Source("PoolSource",
   skipEvents = cms.untracked.uint32(0),
   fileNames = cms.untracked.vstring(
      '/store/data/Commissioning08/BeamHalo/RAW/StuffAlmostToP5_v1/000/061/642/94D3CADF-A47D-DD11-BF3E-000423D94E1C.root'
   )
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.load("CalibCalorimetry.EcalTrivialCondModules.EcalTrivialCondRetriever_cfi")
process.EcalTrivialConditionRetriever.producedEcalWeights = False
process.EcalTrivialConditionRetriever.producedEcalPedestals = False
process.EcalTrivialConditionRetriever.producedEcalIntercalibConstants = False
process.EcalTrivialConditionRetriever.producedEcalIntercalibErrors = False
process.EcalTrivialConditionRetriever.producedEcalGainRatios = False
process.EcalTrivialConditionRetriever.producedEcalADCToGeVConstant = False
process.EcalTrivialConditionRetriever.producedEcalLaserCorrection = False
process.EcalTrivialConditionRetriever.producedChannelStatus = cms.untracked.bool(False)
#es_prefer_EcalChannelStatus = cms.ESPrefer("EcalTrivialConditionRetriever","EcalChannelStatus")

process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.GlobalRuns.ForceZeroTeslaField_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_noesprefer_cff")
process.GlobalTag.globaltag = 'CRUZET4_V5P::All'

process.dumpEv = cms.EDAnalyzer("EventContentAnalyzer")

process.MessageLogger = cms.Service("MessageLogger",
    #suppressInfo = cms.untracked.vstring('ecalEBunpacker'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG')
    ),
    categories = cms.untracked.vstring('EcalMipGraphs'),
    destinations = cms.untracked.vstring('cout')
)

process.TFileService = cms.Service("TFileService",
  fileName = cms.string('ecalDisplaysEvent7691-61642.root'),
  closeFileFast = cms.untracked.bool(True)
)

# EventNumberFilter UserCode/CCEcal/CRUZET2/CaloOnlineTools/EventNumberFilter
#process.load("CaloOnlineTools.EventNumberFilter.eventNumberFilter_cfi")
#process.eventNumberFilter.InterestingEvents = (7691,7995,8282,8551,8853,9116,9414,9694,9999,10295,11971,12261,13073)

# L1 Trigger Filter see https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideL1TriggerFAQ
#process.load("L1TriggerConfig.L1GtConfigProducers.L1GtConfig_cff")
#process.load("L1TriggerConfig.L1GtConfigProducers.startup.startupL1Menu_startup_v3_Unprescaled_cff")
#process.load("L1TriggerConfig.L1ScalesProducers.L1MuTriggerScalesConfig_cff")
#process.load("L1TriggerConfig.L1ScalesProducers.L1MuTriggerPtScaleConfig_cff")
#process.load("L1TriggerConfig.L1GtConfigProducers.L1GtBoardMapsConfig_cff")
#process.load("EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi")
#process.gtDigis = process.l1GtUnpack.clone()
#process.load("L1Trigger.Skimmer.l1Filter_cfi")
# For the time being the only ecal trigger being used is L1_SingleEG2
#process.l1Filter.algorithms = cms.vstring('L1_SingleEG2')
#process.gtDigis.DaqGtInputTag = 'source'
   
process.p = cms.Path(

# gtDigis and l1Filter to filter by EcalTrigger
#   process.gtDigis
#   *
#   process.l1Filter
#   *

# eventNumberFilter to filter specific events
#   process.eventNumberFilter
#   *

# Standard sequence to run ecalDisplaysByEvent
   process.ecalEBunpacker
   *
   process.ecalUncalibHit
   *
   process.ecalRecHit
   *
   process.ecalDisplaysByEvent
)
