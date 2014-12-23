import FWCore.ParameterSet.Config as cms

ggHiNtuplizer = cms.EDAnalyzer("ggHiNtuplizer",
    doGenParticles     = cms.bool(True),
    runOnParticleGun   = cms.bool(False),
    pileupCollection   = cms.InputTag("addPileupInfo"),
    genParticleSrc     = cms.InputTag("hiGenParticles"),
    gsfElectronLabel   = cms.InputTag("ecalDrivenGsfElectrons"),
    recoPhotonSrc      = cms.InputTag("photons"),
    ebRecHitCollection = cms.InputTag("ecalRecHit", "EcalRecHitsEB"),
    eeRecHitCollection = cms.InputTag("ecalRecHit", "EcalRecHitsEE"),
    VtxLabel           = cms.InputTag("hiSelectedVertex"),
)
