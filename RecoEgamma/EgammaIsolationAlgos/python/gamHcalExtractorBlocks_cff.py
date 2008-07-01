import FWCore.ParameterSet.Config as cms

GamIsoHcalFromHitsExtractorBlock = cms.PSet(
    ComponentName = cms.string('EgammaHcalExtractor'),
    DepositLabel = cms.untracked.string(''),
    hcalRecHits = cms.InputTag("hbhereco"),
    minCandEt = cms.double(15.),
    extRadius = cms.double(0.5),
    intRadius = cms.double(0.0),
    etMin = cms.double(-999.0),

    #Following params are use to decide if candidate is isolated
    barrelEcalHits = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    endcapEcalHits = cms.InputTag("ecalRecHit","EcalRecHitsEE"),

    checkIsoExtRBarrel            = cms.double(0.4),
    checkIsoInnRBarrel            = cms.double(0.045),
    checkIsoEtaStripBarrel        = cms.double(0.02),
    checkIsoEtRecHitBarrel        = cms.double(0.08),
    checkIsoEtCutBarrel           = cms.double(8.),

    checkIsoExtREndcap            = cms.double(0.4),
    checkIsoInnREndcap            = cms.double(0.07),
    checkIsoEtaStripEndcap        = cms.double(0.02),
    checkIsoEtRecHitEndcap        = cms.double(0.30),
    checkIsoEtCutEndcap           = cms.double(7.)
)

GamIsoHcalFromTowersExtractorBlock = cms.PSet(
    caloTowers = cms.InputTag("towerMaker"),
    ComponentName = cms.string('EgammaTowerExtractor'),
    intRadius = cms.double(0.0),
    extRadius = cms.double(0.6),
    DepositLabel = cms.untracked.string(''),
    etMin = cms.double(-999.0)
)

