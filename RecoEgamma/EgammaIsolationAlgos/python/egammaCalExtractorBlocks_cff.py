import FWCore.ParameterSet.Config as cms

EgammaIsoEcalSCVetoFromClustsExtractorBlock = cms.PSet(
    ComponentName = cms.string('EgammaEcalExtractor'),
    superClusters = cms.InputTag("egammaSuperClusterMerger"),
    basicClusters = cms.InputTag("egammaBasicClusterMerger"),
    extRadius = cms.double(0.6),
    DepositLabel = cms.untracked.string(''),
    etMin = cms.double(-999.0),
    superClusterMatch = cms.bool(True)
)
EgammaIsoEcalFromClustsExtractorBlock = cms.PSet(
    ComponentName = cms.string('EgammaEcalExtractor'),
    superClusters = cms.InputTag("egammaSuperClusterMerger"),
    basicClusters = cms.InputTag("egammaBasicClusterMerger"),
    extRadius = cms.double(0.6),
    DepositLabel = cms.untracked.string(''),
    etMin = cms.double(-999.0),
    superClusterMatch = cms.bool(False)
)
EgammaIsoEcalFromHitsExtractorBlock = cms.PSet(
    isolationVariable = cms.string('et'),
    tryBoth = cms.bool(True),
    ComponentName = cms.string('EgammaRecHitExtractor'),
    intRadius = cms.double(0.0),
    extRadius = cms.double(0.6),
    endcapRecHits = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    subtractSuperClusterEnergy = cms.bool(False),
    DepositLabel = cms.untracked.string(''),
    detector = cms.string('Ecal'),
    etMin = cms.double(-999.0),
    barrelRecHits = cms.InputTag("ecalRecHit","EcalRecHitsEB")
)
EgammaIsoHcalFromHitsExtractorBlock = cms.PSet(
    ComponentName = cms.string('EgammaHcalExtractor'),
    intRadius = cms.double(0.0),
    extRadius = cms.double(0.6),
    DepositLabel = cms.untracked.string(''),
    hcalRecHits = cms.InputTag("hbhereco"),
    etMin = cms.double(-999.0)
)
EgammaIsoHcalFromTowersExtractorBlock = cms.PSet(
    caloTowers = cms.InputTag("towerMaker"),
    ComponentName = cms.string('EgammaTowerExtractor'),
    intRadius = cms.double(0.0),
    extRadius = cms.double(0.6),
    DepositLabel = cms.untracked.string(''),
    etMin = cms.double(-999.0)
)

