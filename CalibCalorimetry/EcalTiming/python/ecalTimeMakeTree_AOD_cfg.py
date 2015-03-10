import FWCore.ParameterSet.Config as cms

process = cms.Process("TIMECALIBANALYSIS")

# gfworks: to get clustering 
process.load('Configuration/StandardSequences/GeometryExtended_cff')

# Geometry
process.load("Geometry.CaloEventSetup.CaloTopology_cfi")
process.load("Geometry.CaloEventSetup.CaloGeometry_cff")
process.load("Geometry.CaloEventSetup.CaloGeometry_cfi")
process.load("Geometry.EcalMapping.EcalMapping_cfi")
process.load("Geometry.EcalMapping.EcalMappingRecord_cfi")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi") # gfwork: need this?


# Global Tag
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_noesprefer_cff")
#process.GlobalTag.globaltag = 'GR_R_42_V20::All'
#process.GlobalTag.globaltag = 'GR_R_52_V7::All'
#process.GlobalTag.globaltag = 'GR_R_52_V7::All'
#process.GlobalTag.globaltag = 'GR_P_V32::All'
#process.GlobalTag.globaltag = 'GR_P_V33::All'
#process.GlobalTag.globaltag = 'GR_P_V37::All'
#process.GlobalTag.globaltag = 'GR_P_V40::All'
#process.GlobalTag.globaltag = 'FT_R_53_V6::All'
process.GlobalTag.globaltag = 'GR_P_V40::All'

# Trigger
process.load("L1TriggerConfig.L1ScalesProducers.L1MuTriggerScalesConfig_cff")
process.load("L1TriggerConfig.L1ScalesProducers.L1MuTriggerPtScaleConfig_cff")
process.load("L1TriggerConfig.L1GtConfigProducers.L1GtBoardMapsConfig_cff")
process.load("L1TriggerConfig.L1GtConfigProducers.L1GtConfig_cff")
process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.startup.L1Menu_startup2_v2_Unprescaled_cff")
import FWCore.Modules.printContent_cfi
process.dumpEv = FWCore.Modules.printContent_cfi.printContent.clone()

import EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi
process.gtDigis = EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi.l1GtUnpack.clone()


# this is the ntuple producer
process.load("CalibCalorimetry.EcalTiming.ecalTimeTree_cfi")
process.ecalTimeTree.fileName = 'EcalTimeTree'
process.ecalTimeTree.barrelEcalRecHitCollection = cms.InputTag("reducedEcalRecHitsEB","")
process.ecalTimeTree.endcapEcalRecHitCollection = cms.InputTag("reducedEcalRecHitsEE","")
process.ecalTimeTree.barrelBasicClusterCollection = cms.InputTag("hybridSuperClusters","hybridBarrelBasicClusters")
process.ecalTimeTree.endcapBasicClusterCollection = cms.InputTag("multi5x5SuperClusters","multi5x5EndcapBasicClusters")
process.ecalTimeTree.barrelSuperClusterCollection = cms.InputTag("correctedHybridSuperClusters","")
process.ecalTimeTree.endcapSuperClusterCollection = cms.InputTag("correctedMulti5x5SuperClustersWithPreshower","")
process.ecalTimeTree.muonCollection = cms.InputTag("muons")
process.ecalTimeTree.runNum = 999999


process.dumpEvContent = cms.EDAnalyzer("EventContentAnalyzer")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1000))

process.p = cms.Path(
    # process.dumpEvContent  *
    process.ecalTimeTree
    )



process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING')
    ),
    categories = cms.untracked.vstring('ecalTimeTree'),
    destinations = cms.untracked.vstring('cout')
)
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(250)




# GF: some legacy reco files to test; replace w/ collision data
# dbs search --query "find file where dataset=/ExpressPhysics/BeamCommissioning09-Express-v2/FEVT and run=124020" | grep store | awk '{printf "\"%s\",\n", $1}'
process.source = cms.Source(
    "PoolSource",
    skipEvents = cms.untracked.uint32(0),
    
    # a few files from:    /MinimumBias/Commissioning10-GR_R_35X_V7A_SD_EG-v2/RECO
    fileNames = (cms.untracked.vstring(
    #'/store/data/Commissioning10/MinimumBias/RAW-RECO/v9/000/135/494/A4C5C9FA-C462-DF11-BC35-003048D45F7A.root',
    #'/store/data/Run2010A/EG/RECO/v4/000/144/114/EEC21BFA-25B4-DF11-840A-001617DBD5AC.root'
    #'file:AOD.root'
    #'file:/data/franzoni/data/Run2011A-MinimumBias-RECO-PromptReco-v1-run160406-0C132C90-434F-E011-8FF6-003048D2BF1C.root'
    'file:/data/franzoni/data/Run2011A_DoubleElectron_AOD_PromptReco-v4_000_166_946_CE9FBCFF-4B98-E011-A6C3-003048F11C58.root'
    )
                 )
    )
