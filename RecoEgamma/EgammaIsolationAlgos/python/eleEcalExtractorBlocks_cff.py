import FWCore.ParameterSet.Config as cms

EleIsoEcalFromHitsExtractorBlock = cms.PSet(
    ComponentName = cms.string('EgammaRecHitExtractor'),
    DepositLabel = cms.untracked.string(''),
    isolationVariable = cms.string('et'),
    extRadius = cms.double(0.6),
    intRadius = cms.double(0.0),
    intStrip = cms.double(0.0),
    etMin = cms.double(-999.0),
    energyMin = cms.double(0.08),
    subtractSuperClusterEnergy = cms.bool(False),
    tryBoth = cms.bool(True),

    barrelEcalHits = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    endcapEcalHits = cms.InputTag("ecalRecHit","EcalRecHitsEE")

)

