# cloned from SimGeneral/MixingModule/python/trackingTruthProducer_cfi.py

import FWCore.ParameterSet.Config as cms

trackingParticles = cms.PSet(
    accumulatorType = cms.string('TrackingTruthAccumulator'),
    createUnmergedCollection = cms.bool(True),
    createMergedBremsstrahlung = cms.bool(True),
    alwaysAddAncestors = cms.bool(True),
    maximumPreviousBunchCrossing = cms.uint32(9999),
    maximumSubsequentBunchCrossing = cms.uint32(9999),
    vertexDistanceCut = cms.double(0.003),
    simHitCollections = cms.PSet(
#    muon = cms.vstring('MuonDTHits',
#                       'MuonCSCHits',
#                       'MuonRPCHits'),
    tracker = cms.vstring('TrackerHits'),
    pixel = cms.vstring('TrackerHits')
    ),
    simHitLabel = cms.string('famosSimHits'),
    genParticleCollection = cms.InputTag('genParticles'),
    copySimHits = cms.bool(True),
    removeDeadModules = cms.bool(False), # currently not implemented
    useMultipleHepMCLabels = cms.bool(False),
    volumeRadius = cms.double(1200.0),
    volumeZ = cms.double(3000.0),
    ignoreTracksOutsideVolume = cms.bool(False)
    )

trackingParticlesMuons = cms.PSet( # dirty trick needed because simHitLabel is different for muon hits (we had the same problem also with old TP)
    accumulatorType = cms.string('TrackingTruthAccumulator'),
    createUnmergedCollection = cms.bool(True),
    createMergedBremsstrahlung = cms.bool(True),
    alwaysAddAncestors = cms.bool(True),
    maximumPreviousBunchCrossing = cms.uint32(9999),
    maximumSubsequentBunchCrossing = cms.uint32(9999),
    vertexDistanceCut = cms.double(0.003),
    simHitCollections = cms.PSet(
    muon = cms.vstring('MuonDTHits',
                       'MuonCSCHits',
                       'MuonRPCHits'),
 #   tracker = cms.vstring('famosSimHitsTrackerHits'),
 #   pixel = cms.vstring('famosSimHitsTrackerHits')
    ),
    simHitLabel = cms.string('MuonSimHits'),
    genParticleCollection = cms.InputTag('genParticles'),
    copySimHits = cms.bool(True),
    removeDeadModules = cms.bool(False), # currently not implemented
    useMultipleHepMCLabels = cms.bool(False),
    volumeRadius = cms.double(1200.0),
    volumeZ = cms.double(3000.0),
    ignoreTracksOutsideVolume = cms.bool(False)
    )

# cloned from SimGeneral/MixingModule/python/trackingTruthProducerSelection_cfi.py

trackingParticles.select = cms.PSet(
        lipTP = cms.double(1000),
        chargedOnlyTP = cms.bool(True),
        pdgIdTP = cms.vint32(),
        signalOnlyTP = cms.bool(True),
        stableOnlyTP = cms.bool(True), # this is different from the standard setting for FullSim
        minRapidityTP = cms.double(-2.6),
        minHitTP = cms.int32(3),
        ptMinTP = cms.double(0.2),
        maxRapidityTP = cms.double(2.6),
        tipTP = cms.double(1000)
        )

trackingParticlesMuons.select = cms.PSet(
        lipTP = cms.double(1000),
        chargedOnlyTP = cms.bool(True),
        pdgIdTP = cms.vint32(),
        signalOnlyTP = cms.bool(True),
        stableOnlyTP = cms.bool(True), # this is different from the standard setting for FullSim
        minRapidityTP = cms.double(-2.6),
        minHitTP = cms.int32(3),
        ptMinTP = cms.double(0.2),
        maxRapidityTP = cms.double(2.6),
        tipTP = cms.double(1000)
        )
