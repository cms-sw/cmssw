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
    ComponentName = cms.string('EgammaRecHitExtractor'),
    DepositLabel = cms.untracked.string(''),
    barrelRecHits = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    endcapRecHits = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    isolationVariable = cms.string('et'),
    detector = cms.string('Ecal'),
    extRadius = cms.double(0.6),
    intRadius = cms.double(0.0),
    etMin = cms.double(-999.0),
    energyMin = cms.double(0.08),
    subtractSuperClusterEnergy = cms.bool(False),
    tryBoth = cms.bool(True)
)
EgammaIsoHcalFromHitsExtractorBlock = cms.PSet(
    ComponentName = cms.string('EgammaHcalExtractor'),
    DepositLabel = cms.untracked.string(''),
    hcalRecHits = cms.InputTag("hbhereco"),
    extRadius = cms.double(0.6),
    intRadius = cms.double(0.0),
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

