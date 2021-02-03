import FWCore.ParameterSet.Config as cms

trackingParticles = cms.PSet(
    HepMCProductLabel = cms.InputTag("generatorSmeared"),
    accumulatorType = cms.string('TrackingTruthAccumulator'),
    allowDifferentSimHitProcesses = cms.bool(False),
    alwaysAddAncestors = cms.bool(True),
    createInitialVertexCollection = cms.bool(False),
    createMergedBremsstrahlung = cms.bool(True),
    createUnmergedCollection = cms.bool(True),
    genParticleCollection = cms.InputTag("genParticles"),
    ignoreTracksOutsideVolume = cms.bool(False),
    maximumPreviousBunchCrossing = cms.uint32(9999),
    maximumSubsequentBunchCrossing = cms.uint32(9999),
    removeDeadModules = cms.bool(False),
    select = cms.PSet(
        chargedOnlyTP = cms.bool(True),
        intimeOnlyTP = cms.bool(False),
        lipTP = cms.double(1000),
        maxRapidityTP = cms.double(5.0),
        minHitTP = cms.int32(0),
        minRapidityTP = cms.double(-5.0),
        pdgIdTP = cms.vint32(),
        ptMaxTP = cms.double(1e+100),
        ptMinTP = cms.double(0.1),
        signalOnlyTP = cms.bool(False),
        stableOnlyTP = cms.bool(False),
        tipTP = cms.double(1000)
    ),
    simHitCollections = cms.PSet(
        muon = cms.VInputTag(cms.InputTag("g4SimHits","MuonDTHits"), cms.InputTag("g4SimHits","MuonCSCHits"), cms.InputTag("g4SimHits","MuonRPCHits"), cms.InputTag("g4SimHits","MuonGEMHits"), cms.InputTag("g4SimHits","MuonME0Hits")),
        pixel = cms.VInputTag(cms.InputTag("g4SimHits","TrackerHitsPixelBarrelLowTof"), cms.InputTag("g4SimHits","TrackerHitsPixelBarrelHighTof"), cms.InputTag("g4SimHits","TrackerHitsPixelEndcapLowTof"), cms.InputTag("g4SimHits","TrackerHitsPixelEndcapHighTof")),
        tracker = cms.VInputTag()
    ),
    simTrackCollection = cms.InputTag("g4SimHits"),
    simVertexCollection = cms.InputTag("g4SimHits"),
    vertexDistanceCut = cms.double(0.003),
    volumeRadius = cms.double(120.0),
    volumeZ = cms.double(300.0)
)