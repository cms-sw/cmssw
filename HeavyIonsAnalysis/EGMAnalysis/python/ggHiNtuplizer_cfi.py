import FWCore.ParameterSet.Config as cms

ggHiNtuplizer = cms.EDAnalyzer("ggHiNtuplizer",
    doGenParticles = cms.bool(False),
    doSuperClusters = cms.bool(False),
    doElectrons = cms.bool(True),
    doPhotons = cms.bool(True),
    doMuons = cms.bool(True),
    muonPtMin = cms.double(3.0),

    isParticleGun = cms.bool(False),

    doEffectiveAreas = cms.bool(True),
    effAreasConfigFile = cms.FileInPath('HeavyIonsAnalysis/EGMAnalysis/data/EffectiveAreas_94X_v0'),

    superClusters = cms.InputTag("reducedEgamma:reducedSuperClusters"),

    useValMapIso = cms.bool(True),

    doPhoERegression = cms.bool(True),
    doRecHitsEB = cms.bool(False),
    doRecHitsEE = cms.bool(False),
    recHitsEB = cms.untracked.InputTag("reducedEgamma","reducedEBRecHits"),
    recHitsEE = cms.untracked.InputTag("reducedEgamma","reducedEERecHits"),

    pileupSrc = cms.InputTag("slimmedAddPileupInfo"),
    genParticleSrc = cms.InputTag("packedGenParticles"), # use prunedGenParticles for reco::GenParticle objects
    signalGenParticleSrc = cms.InputTag("packedGenParticlesSignal"),
    doPackedGenParticle = cms.bool(True), # use False if prunedGenParticles is genParticleSrc
    vertexSrc = cms.InputTag("offlineSlimmedPrimaryVertices"),
    rhoSrc = cms.InputTag("fixedGridRhoFastjetAll"),
    electronSrc = cms.InputTag("slimmedElectrons"),
    photonSrc = cms.InputTag("slimmedPhotons"),
    muonSrc = cms.InputTag("unpackedMuons"),
    beamSpotSrc = cms.InputTag('offlineBeamSpot'),
    conversionsSrc = cms.InputTag('reducedEgamma:reducedConversions'),

    doPfIso = cms.bool(True),
    particleFlowCollection = cms.InputTag("packedPFCandidates"),
    isPackedPFCandidate = cms.bool(True), # use False if "particleFlow" is particleFlowCollection
)
