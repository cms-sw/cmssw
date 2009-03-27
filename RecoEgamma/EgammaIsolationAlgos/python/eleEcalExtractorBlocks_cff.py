import FWCore.ParameterSet.Config as cms

EleIsoEcalFromHitsExtractorBlock = cms.PSet(
    ComponentName = cms.string('EgammaRecHitExtractor'),
    DepositLabel = cms.untracked.string(''),
    barrelRecHits = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    endcapRecHits = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    isolationVariable = cms.string('et'),
    detector = cms.string('Ecal'),
    minCandEt = cms.double(0.),
    extRadius = cms.double(0.5),
    intRadius = cms.double(0.0),
    intStrip = cms.double(0.0),
    etMin = cms.double(-999.0),
    energyMin = cms.double(0.08),
    subtractSuperClusterEnergy = cms.bool(False),
    tryBoth = cms.bool(True),

    #Following params are use to decide if candidate is isolated
    barrelEcalHits = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    endcapEcalHits = cms.InputTag("ecalRecHit","EcalRecHitsEE"),

    checkIsoExtRBarrel            = cms.double(0.01),
    checkIsoInnRBarrel            = cms.double(0.0),
    checkIsoEtaStripBarrel        = cms.double(0.0),
    checkIsoEtRecHitBarrel        = cms.double(0.08),
    checkIsoEtCutBarrel           = cms.double(10000.),

    checkIsoExtREndcap            = cms.double(0.01),
    checkIsoInnREndcap            = cms.double(0.0),
    checkIsoEtaStripEndcap        = cms.double(0.0),
    checkIsoEtRecHitEndcap        = cms.double(0.30),
    checkIsoEtCutEndcap           = cms.double(10000.)
)

EleIsoEcalSCVetoFromClustsExtractorBlock = cms.PSet(
    ComponentName = cms.string('EgammaEcalExtractor'),
    superClusters = cms.InputTag("egammaSuperClusterMerger"),
    basicClusters = cms.InputTag("egammaBasicClusterMerger"),
    extRadius = cms.double(0.6),
    DepositLabel = cms.untracked.string(''),
    etMin = cms.double(-999.0),
    superClusterMatch = cms.bool(True)
)

EleIsoEcalFromClustsExtractorBlock = cms.PSet(
    ComponentName = cms.string('EgammaEcalExtractor'),
    superClusters = cms.InputTag("egammaSuperClusterMerger"),
    basicClusters = cms.InputTag("egammaBasicClusterMerger"),
    extRadius = cms.double(0.6),
    DepositLabel = cms.untracked.string(''),
    etMin = cms.double(-999.0),
    superClusterMatch = cms.bool(False)
)
