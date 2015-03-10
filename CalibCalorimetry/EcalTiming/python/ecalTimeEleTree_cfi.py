import FWCore.ParameterSet.Config as cms

ecalTimeEleTree = cms.EDAnalyzer("EcalTimeEleTreeMaker",
barrelEcalRecHitCollection = cms.InputTag("reducedEcalRecHitsEB",""),
endcapEcalRecHitCollection = cms.InputTag("reducedEcalRecHitsEE",""),
useRaw = cms.untracked.bool(False),
barrelEcalUncalibratedRecHitCollection = cms.InputTag("ecalRatioUncalibRecHit","EcalUncalibRecHitsEB"),
endcapEcalUncalibratedRecHitCollection = cms.InputTag("ecalRatioUncalibRecHit","EcalUncalibRecHitsEE"),
    # gf set correct cluster producrs
    # gf here SC are used, switch to BC for us
barrelSuperClusterCollection = cms.InputTag("correctedHybridSuperClusters",""),
endcapSuperClusterCollection = cms.InputTag("correctedMulti5x5SuperClustersWithPreshower",""),
barrelBasicClusterCollection = cms.InputTag("correctedHybridSuperClusters",""),
    endcapBasicClusterCollection = cms.InputTag("correctedMulti5x5SuperClustersWithPreshower",""),
barrelClusterShapeAssociationCollection = cms.InputTag("hybridSuperClusters","hybridShapeAssoc"),
endcapClusterShapeAssociationCollection = cms.InputTag("multi5x5BasicClusters","multi5x5EndcapShapeAssoc"),
vertexCollection  = cms.InputTag("offlinePrimaryVertices",""),                patElectrons      = cms.InputTag("patElectrons",""),
eleWorkingPoint   = cms.string('simpleEleId85relIso'),
elePtCut          = cms.double(10),
eleIdCuts         = cms.vint32(5,7),

#   eleIdCuts should be set according to: https://twiki.cern.ch/twiki/bin/viewauth/CMS/SimpleCutBasedEleID#Electron_ID_Implementation_in_Re
#   0: fails
#   1: passes electron ID only
#   2: passes electron Isolation only
#   3: passes electron ID and Isolation only
#   4: passes conversion rejection
#   5: passes conversion rejection and ID
#   6: passes conversion rejection and Isolation
#   7: passes the whole selection

# use a higher Pt treshold for superclusters which are beyond |eta|>2.5   
scHighEtaEEPtCut =  cms.double(30.),
hbTreshold = cms.double(1.),
l1GlobalReadoutRecord = cms.string('gtDigis'),
GTRecordCollection = cms.string('gtDigis'),
runNum = cms.int32(-1),
OutfileName = cms.string('EcalTimeTree'),
TrackAssociatorParameters = cms.PSet(
muonMaxDistanceSigmaX = cms.double(0.0),
muonMaxDistanceSigmaY = cms.double(0.0),
CSCSegmentCollectionLabel = cms.InputTag("cscSegments"),
dRHcal = cms.double(9999.0),
dREcal = cms.double(9999.0),
dRPreshowerPreselection = cms.double(9999.0),
CaloTowerCollectionLabel = cms.InputTag("towerMaker"),
useEcal = cms.bool(True),
usePreshower = cms.bool(True),
dREcalPreselection = cms.double(0.05),
HORecHitCollectionLabel = cms.InputTag("horeco"),
dRMuon = cms.double(9999.0),
crossedEnergyType = cms.string('SinglePointAlongTrajectory'),
propagateAllDirections = cms.bool(True),
muonMaxDistanceX = cms.double(5.0),
muonMaxDistanceY = cms.double(5.0),
useHO = cms.bool(True),
trajectoryUncertaintyTolerance = cms.double(-1),
accountForTrajectoryChangeCalo = cms.bool(False),
DTRecSegment4DCollectionLabel = cms.InputTag("dt4DSegments"),
EERecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
dRHcalPreselection = cms.double(0.2),
useMuon = cms.bool(True),
useCalo = cms.bool(False),
EBRecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
dRMuonPreselection = cms.double(0.2),
truthMatch = cms.bool(False),
HBHERecHitCollectionLabel = cms.InputTag("hbhereco"),
useHcal = cms.bool(True)
    )
)


