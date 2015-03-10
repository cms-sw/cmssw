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
process.GlobalTag.globaltag = 'GR_R_35X_V8A::All'


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



# general basic- and super- clustering sequences
import RecoEcal.EgammaClusterProducers.multi5x5ClusteringSequence_cff

# 3x3 clustering for barrel
process.multi5x5BasicClustersTimePi0Barrel =  RecoEcal.EgammaClusterProducers.multi5x5BasicClusters_cfi.multi5x5BasicClusters.clone(
    # which regions should be clusterized
    doEndcap = cms.bool(False),
    doBarrel = cms.bool(True),

    # gfwork: this is standard, can go away 
    barrelHitProducer = cms.string('ecalRecHit'),
    barrelHitCollection = cms.string('EcalRecHitsEB'),
    endcapHitProducer = cms.string('ecalRecHit'),
    endcapHitCollection = cms.string('EcalRecHitsEE'),
    
    IslandBarrelSeedThr = cms.double(0.5),   # barrelSeedThreshold
    IslandEndcapSeedThr = cms.double(1.0),   # endcapSeedThreshold

    barrelClusterCollection = cms.string('multi5x5BarrelBasicClusters'),
    endcapClusterCollection = cms.string('multi5x5EndcapBasicClusters'),
    clustershapecollectionEE = cms.string('multi5x5EndcapShape'),
    clustershapecollectionEB = cms.string('multi5x5BarrelShape'),
    barrelShapeAssociation = cms.string('multi5x5BarrelShapeAssoc'),
    endcapShapeAssociation = cms.string('multi5x5EndcapShapeAssoc'),
    )


# 3x3 clustering for endcap
process.multi5x5BasicClustersTimePi0Endcap =  RecoEcal.EgammaClusterProducers.multi5x5BasicClusters_cfi.multi5x5BasicClusters.clone(
    # which regions should be clusterized
    doEndcap = cms.bool(True),
    doBarrel = cms.bool(False),

    barrelHitProducer = cms.string('ecalRecHit'),
    barrelHitCollection = cms.string('EcalRecHitsEB'),
    endcapHitProducer = cms.string('ecalRecHit'),
    endcapHitCollection = cms.string('EcalRecHitsEE'),
    
    IslandBarrelSeedThr = cms.double(0.5),              # endcapSeedThreshold
    IslandEndcapSeedThr = cms.double(1.0),             # barrelSeedThreshold

    barrelClusterCollection = cms.string('multi5x5BarrelBasicClusters'),
    endcapClusterCollection = cms.string('multi5x5EndcapBasicClusters'),
    clustershapecollectionEE = cms.string('multi5x5EndcapShape'),
    clustershapecollectionEB = cms.string('multi5x5BarrelShape'),
    barrelShapeAssociation = cms.string('multi5x5BarrelShapeAssoc'),
    endcapShapeAssociation = cms.string('multi5x5EndcapShapeAssoc'),
    )


# super clustering for the ECAL BARREL, staring from multi5x5 3x3 clusters
process.multi5x5SuperClustersTimePi0Barrel =  RecoEcal.EgammaClusterProducers.multi5x5SuperClusters_cfi.multi5x5SuperClusters.clone(
    doBarrel = cms.bool(True),
    doEndcaps = cms.bool(False),
    barrelClusterProducer = cms.string('multi5x5BasicClustersTimePi0Barrel'),
    barrelClusterCollection = cms.string('multi5x5BarrelBasicClusters'),
    endcapClusterProducer = cms.string('multi5x5BasicClustersTimePi0Endcap'),
    endcapClusterCollection = cms.string('multi5x5EndcapBasicClusters')
 )


# super clustering for the ECAL ENDCAP, staring from multi5x5 3x3 clusters
process.multi5x5SuperClustersTimePi0Endcap =  RecoEcal.EgammaClusterProducers.multi5x5SuperClusters_cfi.multi5x5SuperClusters.clone(
    doBarrel = cms.bool(False),
    doEndcaps = cms.bool(True),
    barrelClusterProducer = cms.string('multi5x5BasicClustersTimePi0Barrel'),
    barrelClusterCollection = cms.string('multi5x5BarrelBasicClusters'),
    endcapClusterProducer = cms.string('multi5x5BasicClustersTimePi0Endcap'),
    endcapClusterCollection = cms.string('multi5x5EndcapBasicClusters')
 )




# this is the ntuple producer
process.load("CalibCalorimetry.EcalTiming.ecalTimeTree_cfi")
process.ecalTimeTree.fileName = 'EcalTimeTree'
process.ecalTimeTree.muonCollection = cms.InputTag("muons")
process.ecalTimeTree.runNum = 108645
# gfworks: replathese names
process.ecalTimeTree.barrelSuperClusterCollection = cms.InputTag("multi5x5SuperClustersTimePi0Barrel","multi5x5BarrelSuperClusters")
process.ecalTimeTree.endcapSuperClusterCollection = cms.InputTag("multi5x5SuperClustersTimePi0Endcap","multi5x5EndcapSuperClusters")
process.ecalTimeTree.barrelBasicClusterCollection = cms.InputTag("multi5x5BasicClustersTimePi0Barrel","multi5x5BarrelBasicClusters")
process.ecalTimeTree.endcapBasicClusterCollection = cms.InputTag("multi5x5BasicClustersTimePi0Endcap","multi5x5EndcapBasicClusters")
process.ecalTimeTree.barrelClusterShapeAssociationCollection = cms.InputTag("multi5x5BasicClustersTimePi0Barrel","multi5x5BarrelShapeAssoc")
process.ecalTimeTree.endcapClusterShapeAssociationCollection = cms.InputTag("multi5x5BasicClustersTimePi0Endcap","multi5x5EndcapShapeAssoc") 



process.dumpEvContent = cms.EDAnalyzer("EventContentAnalyzer")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1000))

process.p = cms.Path(
    process.multi5x5BasicClustersTimePi0Barrel *
    process.multi5x5BasicClustersTimePi0Endcap *
    process.multi5x5SuperClustersTimePi0Barrel *
    process.multi5x5SuperClustersTimePi0Endcap *
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




# GF: some legacy reco files to test; replace w/ collision data
# dbs search --query "find file where dataset=/ExpressPhysics/BeamCommissioning09-Express-v2/FEVT and run=124020" | grep store | awk '{printf "\"%s\",\n", $1}'
process.source = cms.Source(
    "PoolSource",
    skipEvents = cms.untracked.uint32(0),

    # a few files from:    /MinimumBias/Commissioning10-GR_R_35X_V7A_SD_EG-v2/RECO
    fileNames = (cms.untracked.vstring(
    #'/store/data/Commissioning10/MinimumBias/RAW-RECO/v9/000/135/494/A4C5C9FA-C462-DF11-BC35-003048D45F7A.root',
    '/store/data/Run2010A/EG/RECO/v4/000/144/114/EEC21BFA-25B4-DF11-840A-001617DBD5AC.root'
     )
                  )
    )



