import FWCore.ParameterSet.Config as cms

ggHiNtuplizer = cms.EDAnalyzer(
    "ggHiNtuplizer",
    doGenParticles     = cms.bool(True),
    runOnParticleGun   = cms.bool(False),
    useValMapIso       = cms.bool(True),
    pileupCollection   = cms.InputTag("addPileupInfo"),
    genParticleSrc     = cms.InputTag("hiGenParticles"),
    gsfElectronLabel   = cms.InputTag("ecalDrivenGsfElectrons"),
    recoPhotonSrc      = cms.InputTag("photons"),
    recoPhotonHiIsolationMap = cms.InputTag("photonIsolationHIProducer"),
    recoMuonSrc        = cms.InputTag("muons"),
    VtxLabel           = cms.InputTag("hiSelectedVertex"),
)
