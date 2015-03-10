import FWCore.ParameterSet.Config as cms

process = cms.Process("TIMECALIBANALYSIS")

# gfworks: to get clustering 
process.load('Configuration/StandardSequences/GeometryExtended_cff')

# Geometry
process.load("Geometry.CaloEventSetup.CaloTopology_cfi")
process.load("Geometry.CaloEventSetup.CaloGeometry_cff")
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi") # gfwork: need this? 
process.load("Geometry.CaloEventSetup.CaloGeometry_cfi")
process.load("Geometry.CommonDetUnit.globalTrackingGeometry_cfi") # gfwork: need this?
process.load("Geometry.EcalMapping.EcalMapping_cfi")
process.load("Geometry.EcalMapping.EcalMappingRecord_cfi")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi") # gfwork: need this?

process.load("Configuration.StandardSequences.MagneticField_38T_cff")

# Global Tag
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_noesprefer_cff")
#process.GlobalTag.globaltag = 'CRAFT_ALL_V12::All'
#process.GlobalTag.globaltag = 'GR_R_35X_V8A::All'
#process.GlobalTag.globaltag = 'GR_R_42_V2::All'
#process.GlobalTag.globaltag = 'GR_P_V22::All'
process.GlobalTag.globaltag = 'GR_P_V27::All'

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

# Raw
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.RawToDigi_Data_cff")
#process.load("RecoLocalCalo.EcalRecProducers.ecalRatioUncalibRecHit_cfi")
process.load("RecoLocalCalo.EcalRecProducers.ecalGlobalUncalibRecHit_cfi")
process.load("RecoLocalCalo.Configuration.ecalLocalRecoSequence_cff")
#process.ecalLocalRecoSequence = cms.Sequence(process.ecalRatioUncalibRecHit*process.ecalDetIdToBeRecovered*process.ecalRecHit+process.ecalPreshowerRecHit)
#process.ecalLocalRecoSequence = cms.Sequence(process.ecalGlobalUncalibRecHit*process.ecalDetIdToBeRecovered*process.ecalRecHit+process.ecalPreshowerRecHit)
#process.ecalRecHit.EEuncalibRecHitCollection = cms.InputTag("ecalGlobalUncalibRecHit","EcalUncalibRecHitsEE")
#process.ecalRecHit.EBuncalibRecHitCollection = cms.InputTag("ecalGlobalUncalibRecHit","EcalUncalibRecHitsEB")
#process.ecalRecHit.EEuncalibRecHitCollection = cms.InputTag("ecalRatioUncalibRecHit","EcalUncalibRecHitsEE")
#process.ecalRecHit.EBuncalibRecHitCollection = cms.InputTag("ecalRatioUncalibRecHit","EcalUncalibRecHitsEB")

process.ecalGlobalUncalibRecHit.doEBtimeCorrection = cms.bool(True)
process.ecalGlobalUncalibRecHit.doEEtimeCorrection = cms.bool(True)

# general basic- and super- clustering sequences
import RecoEcal.EgammaClusterProducers.multi5x5ClusteringSequence_cff

# 3x3 clustering for barrel
#process.multi5x5BasicClustersTimePi0Barrel =  RecoEcal.EgammaClusterProducers.multi5x5BasicClusters_cfi.multi5x5BasicClusters.clone(
#    # which regions should be clusterized
#    doEndcap = cms.bool(False),
#    doBarrel = cms.bool(True),
#
#    # gfwork: this is standard, can go away 
#    barrelHitProducer = cms.string('ecalRecHit'),
#    barrelHitCollection = cms.string('EcalRecHitsEB'),
#    endcapHitProducer = cms.string('ecalRecHit'),
#    endcapHitCollection = cms.string('EcalRecHitsEE'),
#    
#    IslandBarrelSeedThr = cms.double(0.5),   # barrelSeedThreshold
#    IslandEndcapSeedThr = cms.double(1.0),   # endcapSeedThreshold
#
#    barrelClusterCollection = cms.string('multi5x5BarrelBasicClusters'),
#    endcapClusterCollection = cms.string('multi5x5EndcapBasicClusters'),
#    clustershapecollectionEE = cms.string('multi5x5EndcapShape'),
#    clustershapecollectionEB = cms.string('multi5x5BarrelShape'),
#    barrelShapeAssociation = cms.string('multi5x5BarrelShapeAssoc'),
#    endcapShapeAssociation = cms.string('multi5x5EndcapShapeAssoc'),
#    )


# 3x3 clustering for endcap
#process.multi5x5BasicClustersTimePi0Endcap =  RecoEcal.EgammaClusterProducers.multi5x5BasicClusters_cfi.multi5x5BasicClusters.clone(
#    # which regions should be clusterized
#    doEndcap = cms.bool(True),
#    doBarrel = cms.bool(False),
#
#    barrelHitProducer = cms.string('ecalRecHit'),
#    barrelHitCollection = cms.string('EcalRecHitsEB'),
#    endcapHitProducer = cms.string('ecalRecHit'),
#    endcapHitCollection = cms.string('EcalRecHitsEE'),
#    
#    IslandBarrelSeedThr = cms.double(0.5),              # endcapSeedThreshold
#    IslandEndcapSeedThr = cms.double(1.0),             # barrelSeedThreshold
#
#    barrelClusterCollection = cms.string('multi5x5BarrelBasicClusters'),
#    endcapClusterCollection = cms.string('multi5x5EndcapBasicClusters'),
#    clustershapecollectionEE = cms.string('multi5x5EndcapShape'),
#    clustershapecollectionEB = cms.string('multi5x5BarrelShape'),
#    barrelShapeAssociation = cms.string('multi5x5BarrelShapeAssoc'),
#    endcapShapeAssociation = cms.string('multi5x5EndcapShapeAssoc'),
#    )


## super clustering for the ECAL BARREL, staring from multi5x5 3x3 clusters
#process.multi5x5SuperClustersTimePi0Barrel =  RecoEcal.EgammaClusterProducers.multi5x5SuperClusters_cfi.multi5x5SuperClusters.clone(
#    doBarrel = cms.bool(True),
#    doEndcaps = cms.bool(False),
#    barrelClusterProducer = cms.string('multi5x5BasicClustersTimePi0Barrel'),
#    barrelClusterCollection = cms.string('multi5x5BarrelBasicClusters'),
#    endcapClusterProducer = cms.string('multi5x5BasicClustersTimePi0Endcap'),
#    endcapClusterCollection = cms.string('multi5x5EndcapBasicClusters')
# )


## super clustering for the ECAL ENDCAP, staring from multi5x5 3x3 clusters
#process.multi5x5SuperClustersTimePi0Endcap =  RecoEcal.EgammaClusterProducers.multi5x5SuperClusters_cfi.multi5x5SuperClusters.clone(
#    doBarrel = cms.bool(False),
#    doEndcaps = cms.bool(True),
#    barrelClusterProducer = cms.string('multi5x5BasicClustersTimePi0Barrel'),
#    barrelClusterCollection = cms.string('multi5x5BarrelBasicClusters'),
#    endcapClusterProducer = cms.string('multi5x5BasicClustersTimePi0Endcap'),
#    endcapClusterCollection = cms.string('multi5x5EndcapBasicClusters')
# )




# this is the ntuple producer
process.load("CalibCalorimetry.EcalTiming.ecalTimeTree_cfi")
process.ecalTimeTree.fileName = 'EcalTimeTree'
process.ecalTimeTree.muonCollection = cms.InputTag("muons")
process.ecalTimeTree.runNum = 999999
# gfworks: replathese names
#process.ecalTimeTree.barrelClusterShapeAssociationCollection = cms.InputTag("multi5x5BasicClustersTimePi0Barrel","multi5x5BarrelShapeAssoc")
#process.ecalTimeTree.endcapClusterShapeAssociationCollection = cms.InputTag("multi5x5BasicClustersTimePi0Endcap","multi5x5EndcapShapeAssoc") 
# use full rechit collection, while from AOD reducedEcalRecHitsEx collections are assumed
process.ecalTimeTree.barrelEcalRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEB")
process.ecalTimeTree.endcapEcalRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEE")

process.load("RecoVertex.Configuration.RecoVertex_cff")


process.dumpEvContent = cms.EDAnalyzer("EventContentAnalyzer")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(2))

import RecoEcal.Configuration.RecoEcal_cff

process.p = cms.Path(
    process.ecalDigis *
    process.gctDigis *
    process.gtDigis *
    process.gtEvmDigis *
    process.siPixelDigis *
    process.siStripDigis *
    process.offlineBeamSpot *
    process.trackerlocalreco *
    process.recopixelvertexing *
    process.ckftracks *
    process.ecalPreshowerDigis *
    process.ecalLocalRecoSequence *
    process.ecalClusters *
    process.vertexreco *
    #process.dumpEvContent  *
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
process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(100)



# GF: some legacy reco files to test; replace w/ collision data
# dbs search --query "find file where dataset=/ExpressPhysics/BeamCommissioning09-Express-v2/FEVT and run=124020" | grep store | awk '{printf "\"%s\",\n", $1}'
process.source = cms.Source(
    "PoolSource",
    skipEvents = cms.untracked.uint32(0),
    
    fileNames = (cms.untracked.vstring(
    'file:/data/franzoni/data/423_Run2011A-SingleMu-RAW-RECO-WMu-May10ReReco-v1-0000-02367CF3-DB7B-E011-8E9D-0019BB32F1EE.root'
    )                ),
    # drop native rechits and clusters, to be sure only those locally made will be picked up
    inputCommands = cms.untracked.vstring('keep *'
                                          ,'drop EcalRecHitsSorted_*_*_RECO' # drop hfRecoEcalCandidate as remade in this process
                                          , 'drop recoSuperClusters_*_*_RECO' # drop hfRecoEcalCandidate as remade in this process
                                          , 'drop recoCaloClusters_*_*_RECO'
                                          )


    )



