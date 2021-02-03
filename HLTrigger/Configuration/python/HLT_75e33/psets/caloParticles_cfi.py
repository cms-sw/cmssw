import FWCore.ParameterSet.Config as cms

caloParticles = cms.PSet(
    HepMCProductLabel = cms.InputTag("generatorSmeared"),
    MaxPseudoRapidity = cms.double(5.0),
    MinEnergy = cms.double(0.5),
    accumulatorType = cms.string('CaloTruthAccumulator'),
    allowDifferentSimHitProcesses = cms.bool(False),
    doHGCAL = cms.bool(True),
    genParticleCollection = cms.InputTag("genParticles"),
    maximumPreviousBunchCrossing = cms.uint32(0),
    maximumSubsequentBunchCrossing = cms.uint32(0),
    premixStage1 = cms.bool(False),
    simHitCollections = cms.PSet(
        hgc = cms.VInputTag(cms.InputTag("g4SimHits","HGCHitsEE"), cms.InputTag("g4SimHits","HGCHitsHEfront"), cms.InputTag("g4SimHits","HGCHitsHEback"))
    ),
    simTrackCollection = cms.InputTag("g4SimHits"),
    simVertexCollection = cms.InputTag("g4SimHits")
)